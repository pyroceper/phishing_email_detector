from fpdf import FPDF

class PDF(FPDF):

    def create_header(self):
        self.add_page()
        self.set_font('Arial', style='B', size=15)
        self.cell(200, 10, txt = "Phishing Email Test", ln = 1, align = 'C')
        self.line(10, 30, self.w - 10, 30)
        self.set_font('Arial', size=15)
        self.cell(200, 10, txt = "Report", ln = 2, align='C')
        self.ln(10)

        self.set_font('Arial', size=12)
        text = f'Email Text: {self.email}'
        self.cell(200, 10, txt=text, ln = 1, align='L')

    def add_line_break(self):
        self.ln(10)

    def get_email_text(self, email):
        self.email = email

    def add_classification_report(self, model_name, accuracy, classification):
        self.set_font('Arial', size=10)
        text = model_name
        self.cell(200, 10, txt=text, ln = 1, align='C')
        self.line(20, self.get_y(), self.w - 20, self.get_y())
        self.add_line_break()
        text = f'Accuracy: {accuracy}'
        self.cell(200, 10, txt=text, ln = 2, align='L')
        text = f'Email Text classification: {classification}'
        self.cell(200, 10, txt=text, ln = 3, align='L')


    def create_pdf(self, file_name):
        self.output(file_name)

    pass