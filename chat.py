import itchat
from itchat.content import TEXT, NOTE 
itchat.auto_login(hotReload=True) 
types = None
info = None
name = None 
@itchat.msg_register([TEXT])
def receive_info(msg):
    global types
    global info
    global name
    name = msg['FileName']
    types = msg["Type"]
    info = msg["Text"]

 
@itchat.msg_register(NOTE)
def withdraw_info(withdraw_msg):
    if "撤回了一条消息" in withdraw_msg["Text"]:
        if types == "Text":
            itchat.send(msg=withdraw_msg["Text"] +
                        ':' + info, toUserName="filehelper")
 
itchat.run()