const pageHeader = '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">'
    + '<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-MrcW6ZMFYlzcLA8Nl+NtUVF0sA7MsXsP1UyJoMp4YLEuNSfAP+JcXn/tWtIaxVXM" crossorigin="anonymous"></script>'   
     + '<title>Phishing Email Detector</title> '
    + '</head> <body>     <div class="container-sm"> <h1 class="pt-4" style="text-align: center;">Phishing Email Test</h1> '
    + '<hr class="hr"/><br></br>';

const pageFooter = '</div><!--container--></body></html>';

const bootstrapRowStart = '<div class="row">';
const bootstrapRowEnd = '</div>';

const tableHeader = '<table class="table table-striped"><thead><tr><th scope="col">Model</th><th scope="col">Accuracy</th> <th scope="col">Classification</th>'
    + '<th scope="col">Handle</th></tr></thead><tbody>';

const tableFooter = '</table>';

const tableButtons = '<div class="text-center"><form action="/pdf" id="pdf-frm" method="post"><button class="btn btn-success">Export to PDF</button>'
    + '</form><form action="/validate" id="back-frm" class="mt-2" method="post"><button class="btn btn-danger">Back</button></form></div>';

const reportIncorrect = '</td><td><a href="/correction">Report Incorrect</a></td></tr>'

const correctionButtons = '<div class="text-left"><form action="/safe_correction" id="safe-frm" method="post"><button class="btn btn-success">Safe Email</button>'
    + '</form><form action="/p_correction" id="phish-frm" class="mt-2" method="post"><button class="btn btn-danger">Phishing Email</button></form></div></div><!--container-->'
    + '</body></html>';

module.exports = {
    pageHeader: pageHeader,
    pageFooter: pageFooter,
    bootstrapRowStart: bootstrapRowStart,
    bootstrapRowEnd: bootstrapRowEnd,
    tableHeader: tableHeader,
    tableFooter: tableFooter,
    tableButtons: tableButtons,
    reportIncorrect, reportIncorrect,
    correctionButtons: correctionButtons
};

