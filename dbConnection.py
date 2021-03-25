from pymongo import MongoClient
import numpy as np
def connectMongo():
    '''
    connect to a target mongo database
    :return: a mongo database connection
    '''
    myclient = MongoClient("mongodb://124.70.205.181:27017/")
    db = myclient['business']
    db.authenticate('nzp','nzp123456')
    return db
class Mongo():
    '''
    class Mongo represents a mongo conncetion which can do some dataset operations.
    '''
    def __init__(self):
        self.con = MongoClient("mongodb://101.132.98.20:27018/")  # link database
        self.db = self.con["business"]  # choose database
        self.db.authenticate('nzp','nzp123456')
        # self.collection = self.db['store_info']  # choose collection
    def query(self, table):
    # inquery a table without conditions
        collection = self.db[table]
        result = collection.find()
        return result
    def queryLimited(self, table, condition):
    #  inquery a table with a condition
        collection = self.db[table]
        result = collection.find(condition)
        return result
    def delete(self, table, id):
    # delete a data record from a table according to the record id
        collection = self.db[table]
        query = {"_id": id}
        collection.delete_one(query)
if __name__ == '__main__':
    mongo = Mongo()
    # mongo.delete('user_info', '6facc31c7cd111ebb62d00163e1cbf3d')
    myquery = {"data_complete":1}
    users = mongo.queryLimited('user_info', myquery)
    for one in users:
        print(one)
        # print(int(str(one['_id']), 16))