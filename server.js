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
        list = list.slice(0, limit);

        res.json({
            success: true,
            timestamp: new Date().toISOString(),
            count: list.length,
            results: list
        });

    } catch (error) {
        res.json({
            success: false,
            message: "API request failed",
            error: error.toString()
        });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("Server started on port", PORT));
