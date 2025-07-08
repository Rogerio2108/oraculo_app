import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from loaders import *
import tempfile
from langchain.prompts import ChatPromptTemplate

MEMORIA = ConversationBufferMemory()


TIPOS_ARQUIVOS_VALIDOS = ['Site', 'Youtube', 'PDF', 'CSV', 'TXT']

CONFIG_MODELOS = {'Groq': {'modelos':['gemma2-9b-it', 'llama-3.1-8b-instant', 'deepseek-r1-distill-llama-70b'],
                           'chat':ChatGroq},
                  'OpenAI': {'modelos':['gpt-4.1-2025-04-14', 'o1-2024-12-17'],
                             'chat': ChatOpenAI}}



def carrega_arquivos(tipo_de_arquivo,arquivo):
    if tipo_de_arquivo == 'Site':
        documento = carrega_site(arquivo)

    if tipo_de_arquivo == 'Youtube':
        documento = carrega_youtube(arquivo)

    if tipo_de_arquivo == 'PDF':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)

    if tipo_de_arquivo == 'CSV':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_csv(nome_temp)

    if tipo_de_arquivo == 'TXT':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_txt(nome_temp)

    return documento



def carrega_modelo(provedor, modelo, api_key, tipo_de_arquivo, arquivo):

    documento = carrega_arquivos(tipo_de_arquivo,arquivo)


    system_message = '''Você é um assistente amigável chamado Oráculo.
    Você possui acesso às seguintes informações vindas 
    de um documento {}: 
    
    ####
    {}
    ####
    
    Utilize as informações fornecidas para basear as suas respostas.
    
    Sempre que houver $ na sua saída, substita por S.
    
    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
    sugira ao usuário carregar novamente o Oráculo!'''.format(tipo_de_arquivo, documento)

    from langchain.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder
    )

    template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain
    return chat


def pagina_chat():
    st.header('Bem-Vindo ao Oráculo 🤖', divider=True)

    chain = st.session_state.get('chain')

    if chain is None:
        st.error('Carregue o Oraculo')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)

    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    if memoria.buffer_as_messages:
        historico_texto = "\n\n".join(
            [f"{msg.type.upper()}: {msg.content}" for msg in memoria.buffer_as_messages]
        )
        st.download_button(
            label="📥 Baixar histórico da conversa",
            data=historico_texto,
            file_name="historico_oraculo.txt",
            mime="text/plain",
            use_container_width=True
        )

    input_usuario = st.chat_input('Fale com o Oráculo')
    if input_usuario:
        memoria.chat_memory.add_user_message(input_usuario)
        st.chat_message('human').markdown(input_usuario)

        with st.chat_message('ai'):
            resposta_container = st.empty()
            resposta_texto = ""

            chat_model = st.session_state.get('chat')

            if hasattr(chat_model, "stream"):
                for chunk in chat_model.stream(input_usuario):
                    resposta_texto += chunk.content
                    resposta_container.markdown(resposta_texto)
            else:
                resposta = chain.invoke({
                    "input": input_usuario,
                    "chat_history": memoria.buffer_as_messages
                })
                resposta_texto = resposta.content
                resposta_container.markdown(resposta_texto)

        memoria.chat_memory.add_ai_message(resposta_texto)
        st.session_state['memoria'] = memoria

def sidebar():
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS )

        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a URL do site')
        if tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a URL do vídeo')
        if tipo_arquivo == 'PDF':
            arquivo = st.file_uploader('Faça o upload do arquivo PDF', type=['.pdf'])
        if tipo_arquivo == 'CSV':
            arquivo = st.file_uploader('Faça o upload do arquivo CSV', type=['.csv'])
        if tipo_arquivo == 'TXT':
            arquivo = st.file_uploader('Faça o upload do arquivo TXT', type=['.txt'])

    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor dos modelos', CONFIG_MODELOS.keys() )
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
                f'Digite a API key para o provedor {provedor}',
                value= st.session_state.get(f'api_key_{provedor}'))

        st.session_state[f'api_key_{provedor}'] = api_key

    if st.button('Inicializar assistente', use_container_width=True):
        st.session_state['tipo_de_arquivo'] = tipo_arquivo
        st.session_state['arquivo'] = arquivo
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)

    if st.button('Apagar Histórico de conversa', use_container_width=True):
        st.session_state['memoria'] = MEMORIA



def main():
    with st.sidebar:
        sidebar()
    pagina_chat()


if __name__ == '__main__':
    main()