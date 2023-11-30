const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const path = require('path');
const app = express();
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true}));
const htmlGen = require('./html-generator');

app.get('/', (req, res) => {

    res.sendFile(path.join(__dirname, 'web/index.html'));

});

app.post('/validate', async (req, res) => {
    try {
        const {data} = 'test';
        const response = await axios.post('http://127.0.0.1:5000/train', {data});

        if(response.status !== 200) {
            throw new Error('Failed to talk to python microservices');
        }

        const dat = response.data;

        res.sendFile(path.join(__dirname, 'web/validate.html'));

    } catch(error) {
        console.error("Error occurred: ", error);
        res.status(500).json({ error: 'Server error'});
    }
});


app.post('/classify', async (req, res) => {
    try {
        const {email} = req.body;
        console.log(email);
        const response = await axios.post('http://127.0.0.1:5000/predict', {email});

        if(response.status !== 200) {
            throw new Error('Failed to talk to python microservices');
        }

        const data = response.data;
        console.log(data);

        //bootstrap result styling
        const successClass = '"text-primary"';
        const failClass = '"text-danger"';
       
        let nbClass = "";
        let dtClass= "";
        let svmClass = "";

        (data.nb_result == "Phishing Email") ? nbClass = failClass : nbClass = successClass;
        (data.dt_result == "Phishing Email") ? dtClass = failClass : dtClass = successClass;
        (data.svm_result == "Phishing Email") ? svmClass = failClass : svmClass = successClass;

        res.send(htmlGen.pageHeader + htmlGen.bootstrapRowStart + '<h4>' + data.email + '</h4>' + htmlGen.tableHeader
        + ' <tbody><tr><td>Naive Bayes Classifier</td><td>'+ data.nb_phish_prob +'</td><td class='+ nbClass +'>'+ data.nb_result +'</td><td>TODO</td></tr>'
        + '<tr><td>Decision Tree Classifier</td><td>'+ data.dt_phish_prob +'</td><td class='+ dtClass + '>'+ data.dt_result +'</td><td>TODO</td></tr>'
        + '<tr><td>SVM Classifier</td><td>'+ data.svm_phish_prob +'</td><td class='+ svmClass +'>'+ data.svm_result +'</td><td>TODO</td></tr></tbody>'
        + htmlGen.tableFooter + htmlGen.tableButtons + htmlGen.bootstrapRowEnd + htmlGen.pageFooter);
        //res.json(data);

    } catch(error) {
        console.error("Error occurred: ", error);
        res.status(500).json({ error: 'Server error'});
    }
});

app.post('/pdf', async(req, res) => {
    try {
        const {data} = 'test';
        const response = await axios.post('http://127.0.0.1:5000/pdf', {data});

        if(response.status !== 200) {
            throw new Error('Failed to talk to python microservices');
        }

        const dat = response.data;
        console.log(dat);

        res.sendFile(path.join(__dirname, 'web/validate.html'));

    } catch(error) {
        console.error("Error occurred: ", error);
        res.status(500).json({ error: 'Server error'});
    }
});

app.listen(3000, () => {
    console.log("Frontend service test");
});