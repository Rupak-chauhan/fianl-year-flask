import numpy as np
import pandas as pd 
import pickle

def prediction(file_path):
    df = pd.read_csv(file_path)

    df['Handcap'] = df['Handcap'].astype('bool')
    df['Handcap'] = df['Handcap'].map({False:0, True:1})

    df['PatientId'].astype('int64')

    df['Gender'] = df['Gender'].map({'F':0, 'M':1})

    df.set_index('AppointmentID', inplace = True)

    df['PreviousApp'] = df.sort_values(by = ['PatientId','ScheduledDay']).groupby(['PatientId']).cumcount()

    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.strftime('%Y-%m-%d')
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])

    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.strftime('%Y-%m-%d')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

    df['Day_diff'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

    df = df[(df.Age >= 0)]
    df.drop(['ScheduledDay'], axis=1, inplace=True)
    df.drop(['AppointmentDay'], axis=1, inplace=True)
    df.drop('PatientId', axis=1,inplace = True)
    df.drop('Neighbourhood', axis=1,inplace = True)

    # import model and scaler
    model = pickle.load(open("model.pkl",'rb'))
    scaler = pickle.load(open("scaler.pkl",'rb'))

    # scaling
    X_predict = scaler.transform(df)

    y_predicted = model.predict(X_predict)

    count = 0
    for i in y_predicted:
        if i == 1:
            count = count + 1
    total = len(y_predicted)
    no_shows = count
    
    appointment_id = list(df.index)
    no_show = list(y_predicted)
    
    intermediate_dictionary = {'appointment_id':appointment_id, 'no_show':no_show}
    appointmet_noshow_df = pd.DataFrame(intermediate_dictionary)
    
    rslt_df = appointmet_noshow_df[appointmet_noshow_df['no_show'] == 1]
    appointments_missed = rslt_df['appointment_id']
    appointments_missed = list(appointments_missed)
    
    response = {'total':total, 'no_shows':no_shows, 'missed_appointments':appointments_missed}

    return response
