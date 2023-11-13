import mysql.connector

class MySQLAdapter:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
    
    def connect(self):
        """
        Connect to the MySQL database using the provided host, user, password, and database.
        """
        conn = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
            
        )
        return conn
    
    def execute_query(self, query, values=None, update=False,with_dictionary=False,use_cursor=False):
        """
        Execute a MySQL query with optional values and return the result.
        """
        conn = self.connect()
        cursor = conn.cursor(dictionary=with_dictionary)
        if update:
            cursor.execute(query, values)
            result=result=cursor.rowcount
            conn.commit()
        else:
            cursor.execute(query, values)
            result = cursor.fetchall()

        if use_cursor:
            return result,cursor,conn
        else:   
            cursor.close()
            conn.close()
            return result
        
        


