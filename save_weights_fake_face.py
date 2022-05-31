model_json = model.to_json()

with open('model.json','w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')