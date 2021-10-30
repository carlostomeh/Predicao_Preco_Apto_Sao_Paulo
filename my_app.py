# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:46:01 2021

@author: carlo
"""

import streamlit as st
import numpy as np
import pandas as pd
import base64
import joblib
import folium
from streamlit_folium import folium_static
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import geopy.distance 
from streamlit_folium import folium_static



class pre_processing_transform(BaseEstimator, TransformerMixin):
    
    # Classe que realiza Pré Processamento para os problemas 1 e 2
    # Objetivo: Transformador customizado e adaptado ao sklearn
    # Observacoes: Com esta configuração essa classe ganha os atributos "fit","fit_transform" e "transform", nativas do sklearn
    # Parametros: "adc_faturamento_per_capita" -> se True adiciona a coluna "faturamento_per_capita". Default : False
    
    def __init__(self):
        return None
    
    def fit(self,X,y = None):
        return self
    
    def transform(self, X, y = None):
        
        temp = X.copy()
        
        # Create Columns
        temp['Total Rooms']            = temp['Rooms'] + temp['Toilets'] + temp['Suites']
        temp['Total Bedrooms']         = temp['Rooms'] + temp['Suites']
        
        
        # Add District Rate
        district = pd.read_csv('./datasets/district_information.csv', sep=';')
        
        temp = pd.merge(temp, district[['District','district_rate','Latitude_district','Longitude_district']],
                           how='left', on=['District'])

        # Drop object Column District   
        temp.drop(['District'], axis=1, inplace=True)
        
        min_y= -23.8
        max_y= -23.2
        min_x= -46.95
        max_x= -46

            
        # Remove outliers and replace as NAN
        temp['Latitude'][(temp['Latitude'] < min_y )  |
                   (temp['Longitude'] < min_x )  |
                   (temp['Latitude']  > max_y )  |
                   (temp['Longitude'] > max_x )  
                   ]= np.nan

        temp['Longitude'][(temp['Latitude'] < min_y ) |
                           (temp['Longitude'] < min_x )  |
                           (temp['Latitude']  > max_y )  |
                           (temp['Longitude'] > max_x )  
                           ]= np.nan
        
        # Imput NA values with mean point neigborhood
        temp.Latitude = np.where(temp.Latitude.isnull()
                                  , temp.Latitude_district # If Latitude is null replace with Latitude_district
                                  , temp.Latitude # else, keep the original value
                                 )
        temp.Longitude = np.where(temp.Longitude.isnull()
                                          , temp.Longitude_district # If Latitude is null replace with Longitude_district
                                          , temp.Longitude # else, keep the original value
                                         )

        # Drop temp Columns
        temp.drop(['Latitude_district', 'Longitude_district'], axis=1, inplace=True)
        
      
        return temp
    
class pre_processing_transform_cluster(BaseEstimator, TransformerMixin):
    
    # Objetivo: Transformador customizado e adaptado ao sklearn
    # Observacoes: Com esta configuração essa classe ganha os atributos "fit","fit_transform" e "transform", nativas do sklearn
    
    def __init__(self):
        return None
    
    def fit(self,X,y = None):
        return self
    
    def transform(self, X, y = None):
        
        temp = X.copy()
        
        min_y= -23.8
        max_y= -23.2
        min_x= -46.95
        max_x= -46
        
        # Create Columns
        temp['Total_Rooms']            = temp['Rooms'] + temp['Toilets'] + temp['Suites']
        temp['Total_Bedrooms']         = temp['Rooms'] + temp['Suites']
        
        
        # Add District Rate
        district = pd.read_csv('./datasets/district_information.csv', sep=';')
        
        temp = pd.merge(temp, district[['District','district_rate','Latitude_district','Longitude_district']],
                           how='left', on=['District'])

        # Drop object Column District   
        temp.drop(['District'], axis=1, inplace=True)
        
        # Remove outliers and replace as NAN
        temp['Latitude'][(temp['Latitude'] < min_y )  |
                   (temp['Longitude'] < min_x )  |
                   (temp['Latitude']  > max_y )  |
                   (temp['Longitude'] > max_x )  
                   ]= np.nan

        temp['Longitude'][(temp['Latitude'] < min_y ) |
                           (temp['Longitude'] < min_x )  |
                           (temp['Latitude']  > max_y )  |
                           (temp['Longitude'] > max_x )  
                           ]= np.nan
        
        # Imput NA values with mean point neigborhood
        temp.Latitude = np.where(temp.Latitude.isnull()
                                  , temp.Latitude_district # If Latitude is null replace with Latitude_district
                                  , temp.Latitude # else, keep the original value
                                 )
        temp.Longitude = np.where(temp.Longitude.isnull()
                                          , temp.Longitude_district # If Latitude is null replace with Longitude_district
                                          , temp.Longitude # else, keep the original value
                                         )

        # Drop temp Columns
        temp = temp[['Condo', 'Size', 'Total_Rooms', 'Total_Bedrooms', 'Price']]
        
        print(temp.columns)
      
        return temp
    
    
def distance_custom(p1,p2):   
    return geopy.distance.distance(p1, p2).km

def transforma_binario(str_novo):
    if str_novo == "Não": return 0;
    if str_novo == "Sim": return 1;
    
                
def busca_apartamentos_similares(X_all, df): 
    
    '''
    Função que retorna dataframe principal para os 10 apartamentos mais próximos segundo o racional da distancia euclideana com pesos
    Recebe como input o dataframe X_all com todos os dados semi limpos e transforma em:
    
    input_transformed = Dataframe normalizado da requisição do usuario
    df_transformed    = Dataframe X_all limpo e normalizado
    
    Retorna um dataframe com os 10 vizinhos mais proximos por distancia euclideana.
    
    '''
    
    pipeline_novo = Pipeline([  ('Criando as colunas', pre_processing_transform_cluster()),
                            ('escalonando', StandardScaler()) ])

    # transform all data
    df_transformed = pipeline_novo.fit_transform(X_all)

    # user data
    input_transformed = pipeline_novo.transform(df)

                

    # Cria lista vazia
    list_distance = []

    # Itera para as colunas de interesse
    for i in range(0, len(df_transformed)):

        # Calcula as distancias para cada feature
        distancia_1 = (df_transformed[i][0] - input_transformed[0][0]) **2
        distancia_2 = (df_transformed[i][1] - input_transformed[0][1]) **2
        distancia_3 = (df_transformed[i][2] - input_transformed[0][2]) **2
        distancia_4 = (df_transformed[i][3] - input_transformed[0][3]) **2
        distancia_5 = (df_transformed[i][4] - input_transformed[0][4]) **2

        # Cria a medidade de distancia - ( Com pesos )
        distance_temp = (distancia_1 + distancia_2 + distancia_3 + distancia_4 + (2*distancia_5))**(0.5)        

        list_distance.append(distance_temp)


    # Cria coluna de distancia ao centroide

    X_all['distancia_centroide'] = list_distance
    mais_proximos = X_all.dropna(subset=['Latitude', 'Longitude']).sort_values(['distancia_centroide'], ascending = [True])[0:10]

    mais_proximos['Total_Rooms']            = mais_proximos['Rooms'] + mais_proximos['Toilets'] + mais_proximos['Suites']
    mais_proximos['Total_Bedrooms']         = mais_proximos['Rooms'] + mais_proximos['Suites']
    
    return mais_proximos;



def get_data():
    ################################### Data read
    # Unify dataset
    X_all =  pd.read_csv('./datasets/housing_clean.csv', sep=';')

    # Setting a dataset to plot map
    X_all_mapa = X_all.dropna(subset=['Latitude', 'Longitude'])
                
    # District Information
    df_district = pd.read_csv('./datasets/district_information.csv', sep=';')  
    
    return X_all, X_all_mapa, df_district;


    

    
 
def main():
    
    # Configurando paraa barra lateral
    st.set_page_config(layout='wide')
    
    # Titulo
    st.title('Aplicativo - Calculadora de Preço de Imóveis em São Paulo')
    
    st.text('Feito por: @https://github.com/carlostomeh e @https://github.com/LkGd ')
    

    X_all, X_all_mapa, df_district = get_data()
    
    col_esq, col_dir = st.columns(2)
    
    with col_esq:
        st.markdown("Neste aplicativo você pode calcular um preço estimado para o seu apartamento através de caracteristicas fisícas e geográficas. Este foi um trabalho desenvolvido durante o curso de Pós Graduação - Especialização em Ciencia de Dados localizado no Campus Campinas do IFSP. Você pode ler mais sobre este projeto acessando os nossos repósitorio no GitHub.")
        st.empty()
        st.empty()
        st.empty()
    
        st.image("./app_files/img_app.jpg")
        
        
    with col_dir:
        
        
        st.markdown("Para auxiliar na sua busca, disponibilizamos um mapa com todos os dados disponiveis de apartamentos à venda em nossa base de dados, neste mapa é possível realizar filtros para buscar o apartamento ideal, adicione uma distancia máxima deste pontos para o centro de São Paulo descobrir quais apartamentos estão à venda perto de você !")
        
        ########################################################################################################
        #################################################### Mapa 2
        ########################################################################################################
        distance_slider = st.slider('Distancia Máxima até a praça da Sé :', 1,25,5)
        
        
        #def plot_mapa_1(distancia_maxima, Latitude, Longitude):
                
        # Cria colunas que vamos utilizar
        X_all_mapa['Total_Rooms']            = X_all_mapa['Rooms'] + X_all_mapa['Toilets'] + X_all_mapa['Suites']
        X_all_mapa['Total_Bedrooms']         = X_all_mapa['Rooms'] + X_all_mapa['Suites']

        # Cria uma coluna com as distancia, em KM, para o apartamento 
        p1 = (-23.5503099, -46.6363896) 

        X_all_mapa['Distancia_apartamento_selecionado'] = X_all_mapa.apply(lambda x: distance_custom(p1, (x.Latitude,x.Longitude )) , axis=1)

        # Primeiro criamos o objeto m para o mapa
        m2 = folium.Map(location=[-23.5503099, -46.6363896], zoom_start=13)
        
        RESULTADO = X_all_mapa[(X_all_mapa.Distancia_apartamento_selecionado <= distance_slider)].reset_index()

        for i in RESULTADO.index:


            string_test = '<b>Mais detalhes:</b>'\
                                '<br/> <b>Bairro:</b>'+ str(RESULTADO.loc[i].District) +\
                                '<br/> <b>Tamanho: </b>'+ str(RESULTADO.loc[i].Size) +\
                                '<br/> <b>Condominio: </b>'+ str(RESULTADO.loc[i].Condo) +\
                                '<br/> <b>Total Quartos: </b>'+ str(RESULTADO.loc[i].Total_Rooms)  +\
                                '<br/> <b>Total Banheiros: </b>'+ str(RESULTADO.loc[i].Total_Bedrooms)  +\
                                '<br/> <b>Vagas de Estacionamento: </b>'+ str(RESULTADO.loc[i].Parking)



            lat_temp = np.float(RESULTADO.loc[i].Latitude)
            long_temp = np.float(RESULTADO.loc[i].Longitude)


            folium.Marker(location=[lat_temp, long_temp],
                      popup=string_test,
                      tooltip = '<b>Este apartamento esta à venda por R$ '+ str(RESULTADO.loc[i].Price)+'</b>',
                      icon=folium.Icon(color='blue', icon='info-sign'), parse_html=True
            ).add_to(m2)

        folium_static(m2);
    
        
        
    
    st.empty()
    st.empty()
    st.empty()
    
    
    st.header(" Calculadora de Apartamentos ")
    st.empty()
    
    st.markdown("Complete os campos abaixo com as informações sobre o seu apartamento e a calculadora vai estimar o Preço do imóvel." )
        
    # Escolhendo a tabela inicial no sidebar
    bairro = st.selectbox("Em qual bairro a casa esta localizada?", ['--','Alto de Pinheiros', 'Anhanguera',
     'Aricanduva', 'Artur Alvim', 'Barra Funda', 'Bela Vista', 'Belém', 'Bom Retiro', 'Brasilândia', 'Brooklin', 'Brás', 'Butantã',   'Cachoeirinha', 'Cambuci', 'Campo Belo', 'Campo Grande', 'Campo Limpo', 'Cangaíba', 'Capão Redondo', 'Carrão', 'Casa Verde', 'Cidade Ademar', 'Cidade Dutra', 'Cidade Líder', 'Cidade Tiradentes', 'Consolação', 'Cursino', 'Ermelino Matarazzo', 'Freguesia do Ó', 'Grajaú',
 'Guaianazes', 'Iguatemi', 'Ipiranga', 'Itaim Bibi', 'Itaim Paulista', 'Itaquera', 'Jabaquara', 'Jaguaré', 'Jaraguá', 'Jardim Helena',
 'Jardim Paulista', 'Jardim São Luis', 'Jardim Ângela', 'Jaçanã', 'José Bonifácio', 'Lajeado', 'Lapa', 'Liberdade', 'Limão', 'Mandaqui',
 'Medeiros', 'Moema', 'Mooca', 'Morumbi', 'Pari', 'Parque do Carmo', 'Pedreira', 'Penha', 'Perdizes', 'Perus', 'Pinheiros', 'Pirituba',
 'Ponte Rasa', 'Raposo Tavares', 'República', 'Rio Pequeno', 'Sacomã', 'Santa Cecília', 'Santana', 'Santo Amaro', 'Sapopemba',
 'Saúde', 'Socorro', 'São Domingos', 'São Lucas', 'São Mateus', 'São Miguel', 'São Rafael', 'Sé', 'Tatuapé', 'Tremembé', 'Tucuruvi', 'Vila Andrade', 'Vila Curuçá', 'Vila Formosa', 'Vila Guilherme', 'Vila Jacuí', 'Vila Leopoldina', 'Vila Madalena', 'Vila Maria', 'Vila Mariana',
 'Vila Matilde', 'Vila Olimpia', 'Vila Prudente', 'Vila Sônia', 'Água Rasa'])
    
    if bairro != "--":
    
        col1, col2 = st.columns(2)

        with col1:
                       
            # Condo
            condo_slice = st.number_input("Digite o valor mensal atribuida ao Condominio:", min_value=0)
            
            # Size
            size_slice = st.number_input("Digite o tamanho, em metros quadrados, do apartamento:", min_value=0)
            

            
            # Rooms
            quartos = st.number_input("Digite a quantidade total de quartos  do apartamento", min_value=0, max_value = 20)

            # Parking
            vagas_estacionamento = st.number_input("Digite a quantidade de vagas de estacionamento do apartamento:", min_value=0, max_value = 20)
            
            # Toilets
            banheiros = st.number_input("Digite a quantidade total de banheiros no apartamento:", min_value=0, max_value = 20)

            # Suites
            suites = st.number_input("Digite a quantidade total Suites no apartamento:", min_value=0, max_value = 20)




        with col2:
            # Swimming Pool
            piscina = st.radio('O condominio possui piscina?',('Não', 'Sim'))

            # Elevator
            elevador = st.radio('O condominio possui elevador?',( 'Não', 'Sim'))

            # New
            novo = st.radio('O apartamento tem menos de 1 ano?',('Não', 'Sim'))

            # Furnished
            mobilia = st.radio('O apartamento já é mobiliado?',('Não', 'Sim'))
            
            # Latitude e Longitude
            options_geo_location = st.radio('Você conhece as coordenadas geográficas do Apartamento ?',( 'Não', 'Sim'))
            

            if options_geo_location== "Não":
                # Get Lat and Lon of median point in a neighboord                
                Latitude = np.float(df_district['Latitude_district'][df_district['District'] == bairro].values)
                Longitude = np.float(df_district['Longitude_district'][df_district['District'] == bairro].values)
                
            else:
                print("Responda a pergunta acima para liberar as coordenadas")
                
                # Latitude
                Latitude = st.number_input("Latitude:", step=0.000001 ,format="%.6f")
                
                # Longitude 
                Longitude = st.number_input("Longitude", step=0.000001,format="%.6f")
                

        st.empty()


        if (bairro != "--") & (st.button('CALCULAR')):


                dict_variables = {"Condo": [condo_slice],
                                 "Size":   [size_slice],
                                 "Rooms": [quartos],
                                 "Toilets":  [banheiros],
                                 "Suites": [suites],
                                 "Parking":   [vagas_estacionamento],
                                 "Elevator": [transforma_binario(elevador)],
                                 "Furnished":  [transforma_binario(mobilia)] ,
                                 "Swimming Pool":  [transforma_binario(piscina)] ,
                                 "New": [transforma_binario(novo)],
                                 "District":   [bairro],
                                 "Latitude": [Latitude],
                                 "Longitude": [Longitude]
                                 }
                df = pd.DataFrame(dict_variables)


                st.write(df.head())
                
                # Import Models
                teste_random_forest = joblib.load('./best_random_forest.pkl')
                teste_ada = joblib.load('./best_adaboost.pkl')

                # Predict values
                predict_randomforest = teste_random_forest.predict(df)
                predict_ada = teste_ada.predict(df)

                # set price to the predict
                df['Price'] = round(predict_randomforest[0], 0)
                
                
                # Display results
                st.write("O valor do apartamento é :")

                col5, col6 = st.columns(2)

                with col5:
                    st.metric("Para o Modelo RandomForest:",round(predict_randomforest[0], 0))

                with col6:
                    st.metric("Para o Modelo AdaBoost:",round(predict_ada[0],0))

                st.write("Fim")
                
                
                #########################################################################################################
                #########################################################################################################
                
                # if st.button('Gostaria de Verificar Apartamentos Disponíveis?'):

                ########################################################################################################
                ############################################## Mapa 2
                ########################################################################################################
                mais_proximos = busca_apartamentos_similares(X_all, df)

                st.write("Abaixo você pode visualizar a localização do apartamento selecionado, em vermelho, mas também verificar sugestões de apartamentos similares")
                
                
                m1 = folium.Map(location=[Latitude, Longitude], zoom_start=10)

                for i in mais_proximos.index:


                    string_test = '<b>Mais detalhes:</b>'\
                                        '<br/> <b>Bairro:</b>'+ str(mais_proximos.loc[i].District) +\
                                        '<br/> <b>Tamanho: </b>'+ str(mais_proximos.loc[i].Size) +\
                                        '<br/> <b>Condominio: </b>'+ str(mais_proximos.loc[i].Condo) +\
                                        '<br/> <b>Total Quartos: </b>'+ str(mais_proximos.loc[i].Total_Rooms)  +\
                                        '<br/> <b>Total Banheiros: </b>'+ str(mais_proximos.loc[i].Total_Bedrooms)  +\
                                        '<br/> <b>Vagas de Estacionamento: </b>'+ str(mais_proximos.loc[i].Parking)



                    lat_temp = np.float(mais_proximos.loc[i].Latitude)
                    long_temp = np.float(mais_proximos.loc[i].Longitude)


                    folium.Marker(location=[lat_temp, long_temp],
                              popup=string_test,
                              tooltip = '<b>Este apartamento esta à venda por R$ '+ str(mais_proximos.loc[i].Price)+'</b>',
                              icon=folium.Icon(color='blue', icon='info-sign'), parse_html=True
                    ).add_to(m1)    
                
                
                # Adiciona o ponto das informações enviadas pelo usuario
                
                folium.Marker(location=[Latitude, Longitude],
                                    popup='<h3>Localização do Apartamento Selecionado</h3>',
                                    icon=folium.Icon(color='red', icon='info-sign'), 
                                      parse_html=True        ).add_to(m1)
                
                folium_static(m1)

                
                



        else:  
            st.markdown("Quando termina de editar os valores, basta apertar o botão para realizar o calculo:"    )


       

 
if __name__ == '__main__':
    main()