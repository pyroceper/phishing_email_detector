class Database:
    def __init__(self):
        pass

    def save_data(self, file_path, data):
        with open(file_path, 'a') as file:
            file.write(data + '\n')

    def check_save_data(self, file_path, data):
        with open(file_path, 'r') as file:
            for line in file:
                if data.strip() == line.strip():
                    return True
        return False        