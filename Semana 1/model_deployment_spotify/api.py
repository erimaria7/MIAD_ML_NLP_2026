#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title='Spotify Popularity Prediction API',
    description='API para predecir la popularidad de canciones en Spotify')

ns = api.namespace('predict',
     description='Spotify Popularity Predictor')

# Cargar el modelo (Asegúrate de que este archivo incluya el Pipeline de preprocesamiento)
try:
    model = joblib.load('modelo/modelo_popularidad_spotify.joblib')
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# Definición de argumentos (Query Parameters)
parser = api.parser()
parser.add_argument('duration_ms',      type=float, required=True, help='Duración en ms',               location='args')
parser.add_argument('explicit',         type=int,   required=True, help='Explícito (1=sí, 0=no)',       location='args')
parser.add_argument('danceability',     type=float, required=True, help='Bailabilidad (0.0 a 1.0)',     location='args')
parser.add_argument('energy',           type=float, required=True, help='Energía (0.0 a 1.0)',          location='args')
parser.add_argument('key',              type=int,   required=True, help='Tonalidad (-1 a 11)',           location='args')
parser.add_argument('loudness',         type=float, required=True, help='Sonoridad en dB',              location='args')
parser.add_argument('mode',             type=int,   required=True, help='Modalidad (1=mayor, 0=menor)', location='args')
parser.add_argument('speechiness',      type=float, required=True, help='Voz hablada (0.0 a 1.0)',      location='args')
parser.add_argument('acousticness',     type=float, required=True, help='Acústica (0.0 a 1.0)',          location='args')
parser.add_argument('instrumentalness', type=float, required=True, help='Instrumental (0.0 a 1.0)',      location='args')
parser.add_argument('liveness',         type=float, required=True, help='En vivo (0.0 a 1.0)',           location='args')
parser.add_argument('valence',          type=float, required=True, help='Positividad (0.0 a 1.0)',      location='args')
parser.add_argument('tempo',            type=float, required=True, help='Tempo en BPM',                 location='args')
parser.add_argument('time_signature',   type=int,   required=True, help='Firma de tiempo (3 a 7)',      location='args')
parser.add_argument('track_genre',      type=str,   required=True, help='Género musical',               location='args')

resource_fields = api.model('Resource', {
    'popularidad_predicha': fields.Float,
})

@ns.route('/')
class SpotifyApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        # 1. Obtener argumentos
        args = parser.parse_args()

        # 2. Crear el DataFrame con los nombres de columnas EXACTOS del entrenamiento
        # Importante: 'explicit' debe ser booleano para que el modelo no falle
        data = pd.DataFrame([{
            'duration_ms':      args['duration_ms'],
            'explicit':         bool(args['explicit']),  # Corrección: conversión a booleano
            'danceability':     args['danceability'],
            'energy':           args['energy'],
            'key':              args['key'],
            'loudness':         args['loudness'],
            'mode':             args['mode'],
            'speechiness':      args['speechiness'],
            'acousticness':     args['acousticness'],
            'instrumentalness': args['instrumentalness'],
            'liveness':         args['liveness'],
            'valence':          args['valence'],
            'tempo':            args['tempo'],
            'time_signature':   args['time_signature'],
            'track_genre':      args['track_genre']
        }])

        # 3. Realizar predicción
        # Nota: El modelo debe ser un Pipeline que incluya el preprocesamiento de 'track_genre'
        try:
            prediccion = model.predict(data)
            # Aseguramos que el resultado sea un número escalar y esté en el rango 0-100
            resultado = float(np.clip(prediccion[0], 0, 100))
        except Exception as e:
            api.abort(500, f"Error en la predicción: {str(e)}")

        return {
            "popularidad_predicha": resultado
        }, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)