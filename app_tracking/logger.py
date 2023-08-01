from datetime import datetime

class App_Logger:
    def __init__(self,path):
        self.path = path

    def log(self,log_message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        with open(self.path,"a+") as file:
            file.write(str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message +"\n")
