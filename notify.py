import smtplib
import os
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders
from datetime import datetime


def send(subject="Training Update",
                to = [os.environ["GM_ADDRESS"]],
                body = "Hello, here is an update",
                data = {"time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                attach = None
                ):

    fromaddr = os.environ["GM_ADDRESS"]
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, os.environ["GM_PASS"])

    for person in to:
        msg = MIMEMultipart()

        msg['From'] = fromaddr
        msg['To'] = person
        msg['Subject'] = subject

        body = body + "\n\n" + str(data)

        msg.attach(MIMEText(body, 'plain'))

        if attach is not None:
            filename = attach
            attachment = open(os.environ["PWD"] + "/" + filename, "rb")

            part = MIMEBase('application', 'octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

            msg.attach(part)

        text = msg.as_string()
        server.sendmail(fromaddr, person, text)
        server.quit()
