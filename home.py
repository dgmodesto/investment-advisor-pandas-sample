
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import pandas as pd 


def main():
  load_dotenv(find_dotenv())
  # configuração da página
  st.set_page_config(page_title="investment adisor", layout="wide")
  

  st.title("Bem vindo")

  st.text("Acesse o menu invest-advisor no menu ao lado para acessar a funcionaliade")

  
    

if __name__ == '__main__':
  main()