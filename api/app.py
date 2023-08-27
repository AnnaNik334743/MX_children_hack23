from typing import Union
import uvicorn
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, Body, File
import io
import pandas as pd
from pipeline import predict_top_query_string, predict_top_query_csv, prepare_environment
from tqdm import tqdm
import json

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/post-string/")
async def stringpost(input_string: str = Body(..., embed=True),
                     input_topn: int = Body(..., embed=True)):
    result_json_list = predict_top_query_string(input_string, topn=input_topn)
    result_dict = {}
    for i in range(1, 1 + input_topn):
        result_dict[f'result_{i}'] = result_json_list[i - 1]
    return json.dumps(str(result_dict), ensure_ascii=False)


@app.post("/post-csv/")
async def csvpost(input_file: UploadFile = File(...)):
    input_bytes = await input_file.read()
    df = pd.read_csv(io.BytesIO(input_bytes), sep=';')

    predict_result = {'predicted_address': [],
                      'target_building_id': [],
                      'relative_dist': []}

    for address in tqdm(df['address']):
        pred_address, pred_id, pred_sim = predict_top_query_csv(address)
        predict_result['predicted_address'].append(pred_address)
        predict_result['target_building_id'].append(pred_id)
        predict_result['relative_dist'].append(pred_sim)

    for column_name, values_vector in predict_result.items():
        df[column_name] = values_vector

    df = df[['id', 'address','target_building_id']]
    # logic for processing df
    # return bytes(df.to_csv(), encoding='utf-8')
    df.to_csv('updated_file.csv', index=False)
    return FileResponse(path='updated_file.csv', media_type='application/csv', filename='updated_file.csv')


@app.post("/post-csv-streamlit/")
async def csvpost(input_file: UploadFile = File(...)):
    input_bytes = await input_file.read()

    df = pd.read_csv(io.BytesIO(input_bytes), sep=',')

    predict_result = {'predicted_address': [],
                      'target_building_id': [],
                      'relative_dist': []}

    for address in tqdm(df['address']):
        pred_address, pred_id, pred_sim = predict_top_query_csv(address)
        predict_result['predicted_address'].append(pred_address)
        predict_result['target_building_id'].append(pred_id)
        predict_result['relative_dist'].append(pred_sim)

    for column_name, values_vector in predict_result.items():
        df[column_name] = values_vector

    df = df[['id', 'target_building_id']]

    # logic for processing df
    return bytes(df.to_csv(index=False), encoding='utf-8')


@app.get("/update")
def update():
    prepare_environment()


if __name__ == "__main__":
    prepare_environment()
    uvicorn.run(app, host='0.0.0.0', port=8000)
