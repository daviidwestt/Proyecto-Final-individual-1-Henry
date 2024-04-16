from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
import pandas as pd
from torch import cosine_similarity
import uvicorn

app = FastAPI()

developer_df = pd.read_parquet ('C:/Users/CRISTIAN/Desktop/Proyecto final #  1/Proyecto-Final-individual-1-Henry/funciones_parquet/developer.parquet')
users_reviews_dfb = pd.read_parquet ('C:/Users/CRISTIAN/Desktop/Proyecto final #  1/Proyecto-Final-individual-1-Henry/funciones_parquet/users_reviews.parquet')
userdata_df = pd.read_parquet ('C:/Users/CRISTIAN/Desktop/Proyecto final #  1/Proyecto-Final-individual-1-Henry/funciones_parquet/userdata.parquet')
userforgenre_df = pd.read_parquet ('C:/Users/CRISTIAN/Desktop/Proyecto final #  1/Proyecto-Final-individual-1-Henry/funciones_parquet/userforgenre.parquet')
bestdeveloperyear_df = pd.read_parquet ('C:/Users/CRISTIAN/Desktop/Proyecto final #  1/Proyecto-Final-individual-1-Henry/funciones_parquet/bestdeveloperyear.parquet')
reviews_developer_df = pd.read_parquet ('C:/Users/CRISTIAN/Desktop/Proyecto final #  1/Proyecto-Final-individual-1-Henry/funciones_parquet/DeveloperReviewsAnalysis.parquet') 
modelo = pd.read_parquet ('C:/Users/CRISTIAN/Desktop/Proyecto final #  1/Proyecto-Final-individual-1-Henry/funciones_parquet/modelo.parquet')

#endpoint /developer
@app.get("/developer")
def Developer(desarrollador: str):
    '''
    Función que recibe como parametro el nombre de una empresa desarrolladora, y retorna la cantidad de items y el porcentaje de contenido gratuito por año.
    Args:
        desarrollador (str): Nombre de la empresa desarrolladora.
    Returns:
        dict: Diccionario con la cantidad de items y el porcentaje de contenido gratuito por año.
    Ejemplo:
        'Valve'
    '''
    #filtramos el dataframe por desarrollador
    developer = developer_df[developer_df['developer'] == desarrollador]
    #Calcular la cantidad de items por año
    item_año = developer.groupby('release_year')['item_id'].count()
    #filtrar los items gratuitos y contarlos por año
    gratis_año = developer[developer['price'] == 0.0].groupby('release_year')['item_id'].count()
    #calcular el porcentaje de items gratuitos por año
    porcentaje_gratis = ((gratis_año / item_año) * 100).fillna(0).astype(int)
    #y retornamos los resultados en formato de diccionario
    return {
        'cantidad por año': gratis_año.to_dict(),
        'porcentaje gratuito por año': porcentaje_gratis.to_dict()
    }


#endpoint /userdata
@app.get("/userdata")
def UserData(user_id: str):
    '''
    Función que recibe como parametro el ID de un usuario, y retorna el dinero gastado por el usuario, el porcentaje de recomendación en base a las reseñas y cantidad de items.
    Args:
        user_id (str): Identificador único del usuario.
    Returns:
        dict: Diccionario con detalles sobre el usuario, incluyendo dinero gastado, items comprados y porcentaje de recomendaciones.
    Ejemplo:
        'GamekungX'
    '''
    #filtramos el dataframe de reseñas por el ID de usuario
    user = users_reviews_dfb[users_reviews_dfb['user_id'] == user_id]
    #obtenemos el dinero gastado e items comprados del dataframe de datos de usuario
    gasto_dinero = userdata_df[userdata_df['user_id'] == user_id]['price'].iloc[0]
    compra_items = userdata_df[userdata_df['user_id'] == user_id]['items_count'].iloc[0]
    #calculamos la cantidad de recomendaciones y el porcentaje de recomendaciones
    recomendaciones = user['recommend'].sum()
    reviews = len(users_reviews_dfb['user_id'])
    recomendaciones_porcet = ((recomendaciones / reviews) * 100)
    #retornamos los resultados en formato de diccionario
    return {
        'user_id': user_id,
        'dinero gastado': gasto_dinero,
        'items comprados': compra_items.astype(int),
        'Porcentaje de recomendaciones': round(recomendaciones_porcet, 3)
    }

#endpoint /userforgenre
@app.get("/userforgenre")
def UserForGenre(genre: str):
    '''
    Función que recibe como parametro el nombre de un género, y retorna el usuario con más horas jugadas y la acumulación de horas jugadas por año de lanzamiento.
    Args:
        genre (str): Nombre del género.
    Returns:
        dict: Diccionario con el usuario con más horas jugadas y la acumulación de horas jugadas por año.
    Ejemplo: 
        'Indie'
    '''
    #filtramos el dataframe por genero
    df_genre = userforgenre_df[userforgenre_df['genres'] == genre]
    #convertimos las horas jugadas a formato de horas enteras
    df_genre['playtime_forever'] = (df_genre['playtime_forever'] / 60 / 60).astype(int)
    #obtenemos el usuario con mas horas jugadas
    playtime_user_max = df_genre.loc[df_genre['playtime_forever'].idxmax(), 'user_id']
    #calculamos la acomulacion de horas jugadas por año
    playtime_yearly = df_genre.groupby('release_year')['playtime_forever'].sum().reset_index()
     #creamos una lista de diccionario con la informacion de mas horas jugadas por año
    list_playtime = [{'año': int(year), 'Horas': int(hours)}for year, hours in zip(playtime_yearly['release_year'], playtime_yearly['playtime_forever'])]
    #retornamos los resultados en formato de diccionario
    return {
        'usuario con mas horas jugadas para genero' + genre: playtime_user_max, 'Horas jugadas':  playtime_yearly
    }

#endpoint /bestdeveloperyear
@app.get("/bestdeveloperyear")
def BestDeveloperYear(year: int):
    '''
    Función que recibe como parametro un año, y retorna el TOP 3 de desarrolladores con juegos más recomendados por usuarios para el año ingresado.
    Args:
         year (int): Año de lanzamiento de los juegos.    
    Returns:
        dict: Diccionario con el TOP 3 de desarrolladores con juegos más recomendados por usuarios para el año ingresado.
    Ejemplo:
        2017
    '''
    #filtramos el dataframe por año
    año = bestdeveloperyear_df[bestdeveloperyear_df['release_year'] == year]
    #filtrar los juegos recomendados y con analisis de sentimiento positivo
    filtro = año[(año['sentiment'] == 2) & (año['recommend'] == True)]
    #agrupar por desarrollador y sumar los analisis de sentimiento
    developer = filtro.groupby('developer')['sentiment'].sum().reset_index()
    #ordenar en orden desendiente segun la suma de analisis de sentimiento
    ordenado_df = developer.sort_values(by='sentiment', ascending=False)
    #seleccionamos los 3 mejores desarrolladores
    developers_top = ordenado_df.head(3)
    #creamos una lista  de diccionarios con el resultado
    result = [{"Puesto {}: {}".format(i+1, row['developer'])} for i, (_, row) in enumerate(developers_top.iterrows())]
    return result


#endpoint /developerreviewsanalysis
@app.get("/developereviwsanalysis")
def DeveloperReviewsAnalysis(desarrollador: str):
    '''
    Funcióon que recibe como parametro el nombre de una empresa desarrolladora, y retorna la cantidad de reseñas positivas y negativas, descartando las neutras.
    Args:
        desarrollador (str): Nombre de la empresa desarrolladora.
    Returns:
        dict: Diccionario con la cantidad de reseñas positivas y negativas, descartando las neutras.
    Ejemplo:
        'Valve'
    '''
    #seleccionamos los registros de el dataframe y preparan dos contadores especificos para contar las reseñas positivas y negativas
    developer = reviews_developer_df[reviews_developer_df['developer'] == desarrollador]
    positivo = 0
    negativo = 0
    #contamos el número de reseñas positivas y negativas basadas en los valores de la columna 'sentiment' 
    #donde '2' se asume como positivo y cualquier otro valor se cuenta como negativo.
    for sentiment in developer['sentiment']:
        if sentiment == 2:
            positivo += 1
        else: 
            negativo += 1
    #retornamos los resultados en formato de diccionario
    return {
        desarrollador: [f"Negative = {negativo}", f"positive = {positivo}"]
    }

#sistema de recomendacion/recommend_games
@app.get("/games_recommendation")
def games_recommendation(id: int):
    '''
    Función que recibe como parametro el ID de un juego y retorna una lista con los 5 juegos más similares.
    Args:
        id (int): ID del juego.
    Returns:
        dict: Diccionario con los 5 juegos más similares al juego ingresado.
    Ejemplo:
        767400
    '''
    try:
        #filtramos el dataframe por el id del juego
        game = modelo[modelo['item_id'].astype(int) == id]
        #verificamos si el juego con el ID proporcionado existente
        if game.empty:
            raise ValueError(f"no se encontro un juego con el ID {id}.")
        #obtenemos el indice del juego en el dataframe
        idx = game.index[0]
        #tamaño de la muestra para calcular la similitud
        sample_size = 2000
        sample_df = modelo.sample(n=sample_size, random_state=42)
        #calculamos la similitud de coseno entre el juego seleccionado y la muestra
        similitud = cosine_similarity([modelo.iloc[idx, 3:]], sample_df.iloc[:, 3:])
        #obtener la fila de la similitud
        similitud = similitud[0]
        #cramos una lista de tuplas (indice, similitud) y  ordenarla por similitud descendente
        juegmos_similares = [(i, similitud[i]) for i in range(len(similitud)) if i != idx]
        juegmos_similares = sorted(juegmos_similares, key=lambda x: x[1], reverse=True)
        #obtenemos los indices de los juegos mas similares
        juegmos_similares_ind = [i[0] for i in juegmos_similares[:5]]
        #obtenemos los nombres de los juegos mas similiares
        nombre_juegos_similares = sample_df['app_name'].iloc[juegmos_similares_ind].tolist()
        return {'juegos similares': nombre_juegos_similares}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#if __name__ == '__main__':
    #uvicorn.run('myapp:app', host='0.0.0.0', port=8000)