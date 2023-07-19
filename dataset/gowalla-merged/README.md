INTERACTIONs DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file gowalla.inter comprising the locations that users have checked in.
Each record/line in the file has the following fields: user_id, item_id, timestamp, latitude, longitude, num_repeat

user_id: the id of the users and its type is token. 
item_id: the id of the locations and its type is token.
timestamp: the UNIX timestamp of latest checking-in time, and its type is float.
latitude: the latitude of locations, and its type is float.
longitude: the longitude of locations, and its type is float.
num_repeat: the number of times that the user has checked in this location, and its type is float.
