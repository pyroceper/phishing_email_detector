const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const path = require('path');

const app = express();
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true}));

app.get('/', (req, res) => {

    res.sendFile(path.join(__dirname, 'web/index.html'));

});

app.post('/validate', async (req, res) => {
    try {
        const {data} = 'test';
        const response = await axios.post('http://127.0.0.1:5000/nb_train', {data});

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
        const response = await axios.post('http://127.0.0.1:5000/nb_predict', {email});

        if(response.status !== 200) {
            throw new Error('Failed to talk to python microservices');
        }

        const data = response.data;
        res.json(data);

    } catch(error) {
        console.error("Error occurred: ", error);
        res.status(500).json({ error: 'Server error'});
    }

    //const phishingProbability = response.data.phish_prob;
    //res.send(`Email classified as phishing with a probability of ${phishingProbability}`);
});

app.post('/dt', async(req, res) => {
    try {
        const {data} = 'test';
        const response = await axios.post('http://127.0.0.1:5000/dt_train', {data});

        if(response.status !== 200) {
            throw new Error('Failed to talk to python microservices');
        }

        const dat = response.data;

        res.sendFile(path.join(__dirname, 'web/validate2.html'));

    } catch(error) {
        console.error("Error occurred: ", error);
        res.status(500).json({ error: 'Server error'});
    }
});

app.post('/dt_classify', async (req, res) => {
    try {
        const {email} = req.body;
        console.log(email);
        const response = await axios.post('http://127.0.0.1:5000/dt_predict', {email});

        if(response.status !== 200) {
            throw new Error('Failed to talk to python microservices');
        }

        const data = response.data;
        res.json(data);

    } catch(error) {
        console.error("Error occurred: ", error);
        res.status(500).json({ error: 'Server error'});
    }

    //const phishingProbability = response.data.phish_prob;
    //res.send(`Email classified as phishing with a probability of ${phishingProbability}`);
});

app.listen(3000, () => {
    console.log("Frontend service test");
});