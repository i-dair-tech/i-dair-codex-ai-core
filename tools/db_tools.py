import MySQLAdapter as MSA
import os
from dotenv import load_dotenv
from django.conf import settings
import json
load_dotenv()
database = os.environ.get("APP_DB_NAME")
db_username = os.environ.get("APP_DB_USER")
db_password = os.environ.get("APP_DB_PASSWORD")
host = os.environ.get("APP_DB_HOST")
# Create an instance of the MySQL adapter
adapter = MSA.MySQLAdapter(host, db_username, db_password, database)
def get_dataset(id): 
    # Execute the SQL query using the adapter's execute_query method
    rows = adapter.execute_query(f"SELECT file_path FROM dataset WHERE id = {id}")
    filename=adapter.execute_query(f"SELECT file_name FROM dataset WHERE id = {id}")
    filename=filename[0][0].split(".")
    filename=filename[0]
    # Return the rows an the file name
    return rows[0][0] , filename

def save_split_strategy(id_dataset,train, test,seed,shuffle):
    adapter.execute_query(f"UPDATE dataset SET train = {train}, test ={test},seed={seed},shuffle={shuffle} WHERE id ={id_dataset} ",update=True)

def get_trained_model_by_session(id_dataset,model_type,session_id):
    # Execute the SQL query using the adapter's execute_query method
    rows = adapter.execute_query(f"SELECT tm.*, m.name FROM trainedModels as tm JOIN models as m on m.id = tm.id_model WHERE tm.session_id = {session_id} AND tm.id_dataset = {id_dataset} AND m.type='{model_type}' AND tm.train_status='completed'",with_dictionary=True)
    return rows

def get_trained_model(id_dataset,model_type):
    # Execute the SQL query using the adapter's execute_query method
    rows = adapter.execute_query(f"SELECT tm.*, m.name FROM trainedModels as tm JOIN models as m on m.id = tm.id_model WHERE tm.id_dataset = {id_dataset} AND m.type='{model_type}' AND tm.train_status='completed'",with_dictionary=True)
    return rows

def add_trained_model(run_id,id_model,id_dataset,train_progress,session_id, train_status="pending"):
    session_id=int(session_id)
    rows = adapter.execute_query(f"SELECT * FROM trainedModels WHERE id_model = '{id_model}' AND id_dataset='{id_dataset}' AND session_id='{session_id}'",with_dictionary=True)
    if len(rows)==0:
        adapter.execute_query(f"INSERT INTO trainedModels (id_dataset, id_model, run_id,train_status, train_progress,session_id) VALUES('{id_dataset}','{id_model}','{run_id}','{train_status}','{train_progress}','{session_id}')",update=True)   
    else:
        adapter.execute_query(f"UPDATE trainedModels SET run_id ='{run_id}',train_status='{train_status}',train_progress={train_progress}, is_best=0 WHERE id_model = '{id_model}' AND id_dataset='{id_dataset}' AND session_id='{session_id}'",update=True)

def epochs_tracker(session_id,id_model,id_dataset,train_progress,train_status="pending"):
    
    adapter.execute_query(f"UPDATE trainedModels SET train_status='{train_status}',train_progress={train_progress} WHERE id_model = '{id_model}' AND id_dataset='{id_dataset}' AND session_id='{session_id}'",update=True)
 

def save_classes(session_id,id_dataset,id_model,classes):
    rows = adapter.execute_query(f"SELECT * FROM trainedModels WHERE id_model = '{id_model}' AND id_dataset='{id_dataset}' AND session_id='{session_id}' ",with_dictionary=True)
    
    if len(rows)==0:
        adapter.execute_query(f"INSERT INTO trainedModels (id_dataset, id_model, classes , session_id) VALUES('{id_dataset}','{id_model}','{classes}','{session_id}')",update=True)
    else:
        values = (classes, id_model, id_dataset, session_id)
        update_query = (
            "UPDATE trainedModels "
            "SET classes = %s "
            "WHERE id_model = %s AND id_dataset = %s AND session_id = %s"
            )
        adapter.execute_query(update_query, values, update=True)
        #adapter.execute_query(f"UPDATE trainedModels SET classes='{classes}'  WHERE id_model = {id_model} AND id_dataset='{id_dataset}' AND session_id= '{session_id}'",update=True)

def get_dataset_data(id): 
    # Execute the SQL query using the adapter's execute_query method
    rows = adapter.execute_query(f"SELECT * FROM dataset WHERE id = {id}",with_dictionary=True)
    return rows

def edit_trained_model(session_id,id_model,id_dataset,task,target,n_iter):
    rows = adapter.execute_query(f"SELECT * FROM trainedModels WHERE id_model = '{id_model}' AND id_dataset='{id_dataset}' AND '{session_id}'",with_dictionary=True)
    if len(rows)==0:
        adapter.execute_query(f"INSERT INTO trainedModels (id_dataset, id_model, task,target,nbr_iteration,session_id) VALUES('{id_dataset}','{id_model}','{task}','{target}','{n_iter}','{session_id}')",update=True)   
    else:
        adapter.execute_query(f"UPDATE trainedModels SET task='{task}',target='{target}',nbr_iteration='{n_iter}'  WHERE id_model = '{id_model}' AND id_dataset='{id_dataset}' AND session_id='{session_id}'",update=True)

def get_all_trained_model_by_dataset(id_dataset):
    rows = adapter.execute_query(f"SELECT tm.*, m.name,u.email, ts.session_name FROM trainedModels as tm JOIN models as m on m.id = tm.id_model JOIN trainingSession as ts on ts.id = tm.session_id  JOIN user as u on u.id = ts.user_id   WHERE tm.id_dataset = {id_dataset} AND tm.train_status='completed' ORDER BY tm.created_at DESC",with_dictionary=True)
    return rows

def create_train_session(model_id,id_dataset,user_id,task,target,params,session_name):
   row,cursor,conn= adapter.execute_query(f"INSERT INTO trainingSession (user_id, status, session_name) VALUES('{user_id}','pending','{session_name}')",update=True,use_cursor=True)  
   cursor.close()
   conn.close()
   i=0
   for id in model_id:
        nbr_iteration=params[i]["n_iter"]
        adapter.execute_query(f"INSERT INTO trainedModels (id_dataset, id_model, train_status, train_progress,task,target,session_id,nbr_iteration) VALUES('{id_dataset}','{id}','pending','0','{task}','{target}','{cursor.lastrowid}','{nbr_iteration}')",update=True)   
        i=i+1
   return cursor.lastrowid

def change_train_session_status(session_id,status):
   row= adapter.execute_query(f"UPDATE trainingSession SET status='{status}' WHERE id='{session_id}'",update=True)  
   return row

def get_all_trained_model_in_progress(user_id):
    rows = adapter.execute_query(f"SELECT tm.*, m.name, ts.user_id, u.email, ts.session_name FROM trainingSession as ts  JOIN trainedModels as tm on ts.id = tm.session_id JOIN models as m on m.id = tm.id_model JOIN user as u on u.id = ts.user_id WHERE  ts.status='pending' AND u.id='{user_id}' ORDER BY tm.created_at DESC",with_dictionary=True)
    return rows


def get_all_trained_model_in_progress_for_group(id_group):
    rows = adapter.execute_query(f"SELECT tm.*, m.name, ts.user_id, u.email, ts.session_name FROM trainingSession as ts  JOIN trainedModels as tm on ts.id = tm.session_id JOIN models as m on m.id = tm.id_model JOIN user as u on u.id = ts.user_id JOIN dataset as ds on ds.id = tm.id_dataset WHERE  ts.status='pending' AND ds.id_group='{id_group}' ORDER BY tm.created_at DESC",with_dictionary=True)
    return rows

def get_user_by_email(email):
    row= adapter.execute_query(f"SELECT * FROM user WHERE email = '{email}'",with_dictionary=True)
    if len(row)==0:
        return 0
    else:
        return row[0]['id']

def get_all_trained_model_in_progress_by_user(user_id):
    rows = adapter.execute_query(f"SELECT tm.*, m.name, ts.user_id, u.email, ts.session_name, ts.status as sessionStatus, ts.message as errorCause FROM trainingSession as ts  JOIN trainedModels as tm on ts.id = tm.session_id JOIN models as m on m.id = tm.id_model JOIN user as u on u.id = ts.user_id WHERE  ts.user_id='{user_id}' AND ts.is_seen='0' ORDER BY tm.created_at DESC",with_dictionary=True)
    return rows

def get_completed_trained_model_by_user(user_id):
    # Execute the SQL query using the adapter's execute_query method
    rows = adapter.execute_query(f"SELECT tm.*, m.name,ts.user_id,tm.session_id, ts.session_name , ts.status as sessionStatus FROM trainedModels as tm JOIN models as m on m.id = tm.id_model  JOIN trainingSession as ts on ts.id = tm.session_id  JOIN user as u on u.id = ts.user_id WHERE  ts.user_id='{user_id}' AND ts.status != 'pending' and ts.is_seen='0' and tm.train_status='completed' ORDER BY tm.created_at DESC",with_dictionary=True)
    return rows

def change_seen_status_of_session(session_id):
    adapter.execute_query(f"UPDATE trainingSession SET is_seen = 1 WHERE id ={session_id} ",update=True)
    return True


def get_default_hyperparameter(user_id):
    models= adapter.execute_query(f"SELECT id,name,type,hyperparameters,available_feature_selection FROM models",with_dictionary=True)
    for model in models:

        user_preferences=adapter.execute_query(f"SELECT hyperparameter FROM preferences WHERE user_id = {user_id} AND model_id = {model['id']}",
                                               with_dictionary=True)
        if user_preferences:
            user_preferences=json.loads(str(user_preferences[0]["hyperparameter"]))
            model["hyperparameters"]=user_preferences
        else:
            hp=json.loads(str(model["hyperparameters"]))
            model["hyperparameters"]=hp

    return models  
    
def get_original_hyperparameters(model_id):
    model= adapter.execute_query(f"SELECT hyperparameters FROM models where id={model_id}",with_dictionary=True)
    hp=json.loads(str(model[0]["hyperparameters"]))
    return hp

def delete_preference(user_id,model_id):
    adapter.execute_query(f"DELETE FROM `preferences` WHERE user_id={user_id} AND model_id={model_id}",update=True)


def save_params(user_id,model_id,params):
 
    user_preferences=adapter.execute_query(f"SELECT hyperparameter FROM preferences WHERE user_id = {user_id} AND model_id = {model_id}",
                                               with_dictionary=True)
    if user_preferences:
        adapter.execute_query(f"UPDATE preferences SET hyperparameter= '{params}' WHERE user_id = {user_id} AND model_id = {model_id}",update=True)
    else:
        adapter.execute_query(f"INSERT INTO preferences(user_id, model_id, hyperparameter) VALUES ({user_id}, {model_id}, '{params}')",update=True)  

   
def get_preferences(user_id,model_id): 
    # Execute the SQL query using the adapter's execute_query method
    rows = adapter.execute_query(f"SELECT * FROM preferences WHERE user_id = {user_id} AND model_id = {model_id}",with_dictionary=True)
    if rows:
        return rows[0]["hyperparameter"]   
    else:
        models= adapter.execute_query(f"SELECT *  FROM models WHERE id = {model_id}",with_dictionary=True)
        return  models[0]["hyperparameters"]

def group_existence_check(user_id=None,privilege=None,group_name=None,group_id=None):
    if group_id==None:
        rows = adapter.execute_query(f"""
        SELECT ug.user_id,ug.group_id,ug.privilege,g.name
        FROM `user-groups` ug
        JOIN `groups` g ON ug.group_id = g.id
        WHERE ug.user_id = {user_id}
        AND g.name = '{group_name}'
        AND ug.privilege = '{privilege}'
        """, with_dictionary=True)
    elif group_name==None and user_id!=None:
        rows = adapter.execute_query(f"""
        SELECT ug.user_id,ug.group_id,ug.privilege,g.name
        FROM `user-groups` ug
        JOIN `groups` g ON ug.group_id = g.id
        WHERE ug.user_id = {user_id}
        AND ug.group_id = '{group_id}'
        """, with_dictionary=True)  
    else:
        rows = adapter.execute_query(f"SELECT * from `user-groups` WHERE group_id={group_id}")
    if rows:
        return True
    else:
        return False
    
def insert_user_in_group(user_id,privilege,group_name=None,group_id=None,return_group_id=False):
    if group_id==None:
        _,cursor,conn=adapter.execute_query(f"INSERT INTO `groups` (name) VALUES ('{group_name}')",update=True,use_cursor=True)
        group_id=cursor.lastrowid
        cursor.close()
        conn.close()
    adapter.execute_query(f"INSERT INTO `user-groups`  (user_id,group_id,privilege) VALUES ({user_id},{group_id},'{privilege}')",update=True)
    if return_group_id:
        return group_id


def email_existence_check(email):
    rows=adapter.execute_query(f"SELECT * FROM `user` WHERE email='{email}'",with_dictionary=True)
    if rows:
        return True,rows[0]["id"]
    else:
        return False, None
    
def get_user_groups_ids (user_id):
    rows=adapter.execute_query(f"SELECT group_id FROM `user-groups` WHERE user_id={user_id}",with_dictionary=True)
    return rows

def get_groups_user_info(group_ids):
    rows=adapter.execute_query(f"""
        SELECT ug.user_id, ug.group_id,g.name,ug.privilege
        FROM `user-groups` ug
        JOIN `groups` g ON ug.group_id = g.id
        WHERE ug.group_id IN ({group_ids})
        """ ,with_dictionary=True)
    return rows

def get_user_email(user_id):
    row= adapter.execute_query(f"SELECT email FROM user WHERE id = {user_id}",with_dictionary=True)
    return row[0]['email']

def delete_old_members(group_id):
    adapter.execute_query(f"DELETE FROM `user-groups` WHERE group_id={group_id} AND privilege='member'",update=True)
def assigned_dataset_validation(dataset_id,group_id):
    rows=adapter.execute_query(f"SELECT id_group FROM `dataset` WHERE id_group = {group_id} AND id={dataset_id}",with_dictionary=True)
    if rows: 
        return True
    else:
        return False
def assign_dataset_to_group(dataset_id,group_id):
    adapter.execute_query(f"UPDATE `dataset` SET id_group = {group_id} WHERE id={dataset_id}",update=True)

def get_not_completed_trained_model_by_user(user_id):
    # Execute the SQL query using the adapter's execute_query method
    rows = adapter.execute_query(f"SELECT tm.*, m.name,tm.session_id, ts.session_name FROM trainedModels as tm JOIN models as m on m.id = tm.id_model  JOIN trainingSession as ts on ts.id = tm.session_id  WHERE  (ts.status='Failed' OR ts.status='canceled') and ts.is_seen='0' and tm.train_status!='completed' ORDER BY tm.created_at DESC",with_dictionary=True)
    return rows
def update_failed_session(session_id,message):
    
    values = (message, session_id)
    update_query=(
                    "UPDATE `trainingSession`" 
                    "SET message = %s"
                    "WHERE id=%s"
                )
    
    adapter.execute_query(update_query,values,update=True)
    
def get_trained_model_by_id(ids):
    if isinstance(ids,int):
        rows=adapter.execute_query(f"""SELECT * 
                                   FROM trainedModels as tm 
                                   JOIN models as m on m.id = tm.id_model 
                                   JOIN trainingSession as ts on ts.id = tm.session_id  
                                   JOIN user as u on u.id = ts.user_id   
                                   WHERE tm.id ={ids}""",
                                   with_dictionary=True)
    else:
        rows=adapter.execute_query(f"""SELECT tm.id , tm.run_id, ts.session_name, m.name 
                                FROM `trainedModels` tm 
                                JOIN models m ON tm.id_model=m.id 
                                JOIN trainingSession ts ON tm.session_id = ts.id 
                                WHERE tm.id IN ({ids})""",
                                with_dictionary=True)
    
    return rows

def delete_trained_models(trained_models):
    adapter.execute_query(f"DELETE FROM `trainedModels` WHERE id IN ({trained_models})",update=True)

def delete_training_session(all_training_sessions):
    adapter.execute_query(f"DELETE FROM `trainingSession` WHERE id NOT IN ( "
    f"SELECT DISTINCT tm.session_id FROM `trainedModels` AS tm "
    f"WHERE tm.session_id IN ({all_training_sessions})"
    f") AND id IN ({all_training_sessions})",update=True)
    
def delete_trained_models_by_dataset(id_dataset):
    adapter.execute_query( f""" DELETE trainedModels,trainingSession
                               from trainedModels
                               INNER JOIN trainingSession on trainedModels.session_id=trainingSession.id
                               WHERE trainedModels.id_dataset={id_dataset}
                            """
                            ,update=True
                          )
def delete_dataset(id_dataset):
    adapter.execute_query(f"DELETE FROM `dataset` WHERE id = {id_dataset}",update=True)