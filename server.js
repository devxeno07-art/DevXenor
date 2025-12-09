const express = require('express');
const fetch = require('node-fetch');
const cors = require('cors');

const app = express();
app.use(cors());

app.get("/", async (req, res) => {
    const limit = parseInt(req.query.limit) || 10;
    const apiUrl = `https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json?ts=${Date.now()}`;

    try {
        const response = await fetch(apiUrl);
        const data = await response.json();
        let list = data?.data?.list || [];
        res.json({
            success: true,
            timestamp: new Date().toISOString(),
            count: list.length,
            results: list.slice(0, limit)
        });
    } catch (err) {
        res.json({
            success: false,
            message: "API request failed",
            error: err.toString()
        });
    }
});

// ❗Export handler for Vercel
module.exports = (req, res) => app(req, res);
