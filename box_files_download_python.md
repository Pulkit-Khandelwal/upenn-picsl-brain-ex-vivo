# Here we will setup the Box devloper account and then see how to downlaod files and folders automatically
### Author: Pulkit Khandelwal


##### Locate the Box Developer console on the bottom left
<div align="center">
         <img src="https://github.com/Pulkit-Khandelwal/picsl-brain-ex-vivo/blob/main/files/box-dev-1.png" style="width:25%; height:10%">
      </a>
</div>

##### Click on create new app
<div align="center">
         <img src="https://github.com/Pulkit-Khandelwal/picsl-brain-ex-vivo/blob/main/files/box-dev-2-1.png" style="width:100%; height:100%">
      </a>
</div>

##### Enter a name for your app
<div align="center">
         <img src="https://github.com/Pulkit-Khandelwal/picsl-brain-ex-vivo/blob/main/files/box-dev-2.png" style="width:50%; height:40%">
      </a>
</div>


##### Have the following settings and then create the app
<div align="center">
         <img src="https://github.com/Pulkit-Khandelwal/picsl-brain-ex-vivo/blob/main/files/box-dev-3.png" style="width:50%; height:40%">
      </a>
</div>

##### Now, click on configuration and generate a developer token. You will need the Developer Token, Client ID and Client Secret later
<div align="center">
         <img src="https://github.com/Pulkit-Khandelwal/picsl-brain-ex-vivo/blob/main/files/box-dev-4.png" style="width:50%; height:40%">
      </a>
</div>


###### Let's write some simple Python code for locating and downloading your files and folders to a specific folder

You will need to download ```boxsdk``` library from [here](https://github.com/box/box-python-sdk) via ```pip3 install boxsdk```. You will need the ```Developer Token, Client ID and Client Secret``` from above.


```
# Imports and authentication
import os
from boxsdk import OAuth2, Client

auth = OAuth2(
    client_id='GET_FROM_THE_DEV_ACCOUNT',
    client_secret='GET_FROM_THE_DEV_ACCOUNT',
    access_token='GET_FROM_THE_DEV_ACCOUNT',
)
client = Client(auth)

user = client.user().get()
print(f'The current user ID is {user.id}')
```

```
# Try to locate the files and folders in your box folder and get the corresponding file ID.
# Folder 111442091060 is named "Hemisphere_segmentation" in Pulkit's box account (might be different for you)
# Folder 112217518064 is named "subjects" which I need in Pulkit's box account (might be different for you)

root_folder = client.root_folder().get()
items = root_folder.get_items()
for item in items:
    print('{0} {1} is named "{2}"'.format(item.type.capitalize(), item.id, item.name))
```

```
parent_dir = '/data/pulkit/exvivo_data_revised'




```
