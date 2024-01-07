from fastapi import FastAPI, Query
from pycaret.classification import *
import pandas as pd


loaded_model = load_model('/mnt/c/Users/lhmedina/Downloads/DataKnow/Prueba_Tecnica/Punto 6/API/Model_Prod_P6/model_prod')


app = FastAPI()
@app.post("/convert_to_dataframe/")
async def convert_to_dataframe(
    INGRESOS: int = Query(..., description="Ingrese el valor de INGRESOS", example=11000000),
    EGRESOS: int = Query(..., description="Ingrese el valor de EGRESOS", example=4500000),
    Dist_Mean_NAL: float = Query(..., description="Ingrese el valor de Dist_Mean_NAL", example=614.039978),
    Dist_sum_NAL: float = Query(..., description="Ingrese el valor de Dist_sum_NAL", example=1228.069946),
    FECHA_VIN: int = Query(..., description="Ingrese el valor de FECHA_VIN ", example=19890814),
    FECHA: int = Query(..., description="Ingrese el valor de FECHA ", example=20150506),
    OFICINA_VIN: int = Query(..., description="Ingrese el valor de OFICINA_VIN", example=961),
    VALOR: float = Query(..., description="Ingrese el valor de VALOR", example=143202.656250),
    HORA_AUX: int = Query(..., description="Ingrese el valor de HORA_AUX ", example=20),
    EDAD: int = Query(..., description="Ingrese el valor de EDAD ", example=56)
):
    data = {
        'INGRESOS': INGRESOS,
        'EGRESOS': EGRESOS,
        'Dist_Mean_NAL': Dist_Mean_NAL,
        'Dist_sum_NAL': Dist_sum_NAL,
        'FECHA_VIN': FECHA_VIN,
        'FECHA': FECHA,
        'OFICINA_VIN': OFICINA_VIN,
        'VALOR': VALOR,
        'HORA_AUX': HORA_AUX,
        'EDAD': EDAD
    }

    df = pd.DataFrame([data])
    predictions = predict_model(loaded_model, data = df)
    if predictions['prediction_label'].loc[0] == 0:
        mensaje_prediccion = '0 --> NO FRAUDE'
    else:
        mensaje_prediccion = '1 --> FRAUDE'
  

    return {
        #"dataframe": df.to_dict(orient='records'),
        "Prediccion": mensaje_prediccion
    }
