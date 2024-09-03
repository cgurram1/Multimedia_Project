import time
import mysql.connector
import Generator
import ResnetGeneratorTraining

def Train(dataset):
    tic = time.time()
    # create a connection
    connection = mysql.connector.connect(host='localhost',username='root',password='GURRAMgckr@1998',database='Feature_Descriptors')
    my_cursor = connection.cursor()
    table_name = 'Final_Feature_Descriptor'
    query = f"SHOW TABLES LIKE '{table_name}'"
    my_cursor.execute(query)
    result = my_cursor.fetchone()
    if result:
        print(f"{table_name} table already present")
        exit(0)
    else:
        query = f"CREATE TABLE {table_name} (image_Id int,label varchar(255),Feature_Descriptor longtext,HoG_Values longtext, AVG_Pool longtext, layer3 longtext, FC longtext);"
        my_cursor.execute(query)
        # iterate over all the images, generate the vector, convert them to strings and commit the connection
        for image_Id in range(8677):
            print(image_Id)
            img,label = dataset[image_Id]
            color_mom = Generator.Color_Moments(img,0)
            hog = Generator.HoG(img,0)
            avg_pool,layer3,fc_vector = ResnetGeneratorTraining.allLayers(img,image_Id)

            string_list1 = [str(float_num) for float_num in color_mom]
            separator = ', '
            result_string1 = separator.join(string_list1)

            string_list2 = [str(float_num) for float_num in hog]
            separator = ', '
            result_string2 = separator.join(string_list2)

            string_list3 = [str(float_num) for float_num in avg_pool]
            separator = ', '
            result_string3 = separator.join(string_list3)

            string_list4 = [str(float_num) for float_num in layer3]
            separator = ', '
            result_string4 = separator.join(string_list4)

            string_list5 = [str(float_num) for float_num in fc_vector]
            separator = ', '
            result_string5 = separator.join(string_list5)

            insert_query = f"INSERT INTO {table_name} (image_Id, label, Feature_Descriptor,HoG_Values, AVG_Pool, layer3, FC) VALUES (%s, %s, %s, %s, %s, %s, %s);"
            data = (image_Id, label, result_string1,result_string2,result_string3,result_string4,result_string5)
            my_cursor.execute(insert_query, data)
            if(image_Id % 1000 == 0):
                connection.commit()
        connection.commit()
        connection.close()
        toc = time.time()
        print("Finished")
        print(1000*(toc - tic))