require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
app.use(cors());
app.use(express.json());

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ 
    model: "gemini-3.5-flash-lite",
    systemInstruction: "You are a professional AI Personal Finance Assistant for a FinTech app. You will receive the user's financial data alongside their prompt. Analyze their account balances and recent transactions to answer their questions accurately. Provide actionable financial advice. Keep your responses concise and strictly related to finance."
});

// Mock database for the competition
const mockUserData = {
    accounts: { 
        checking: 2450.00, 
        savings: 8500.00 
    },
    recentTransactions: [
        { date: "2026-10-14", merchant: "Whole Foods", amount: 120.50, category: "Groceries" },
        { date: "2026-10-15", merchant: "Uber", amount: 15.00, category: "Transport" },
        { date: "2026-10-16", merchant: "Starbucks", amount: 6.50, category: "Dining" },
        { date: "2026-10-16", merchant: "Netflix", amount: 15.99, category: "Entertainment" }
    ]
};

app.post('/api/chat', async (req, res) => {
    try {
        const { message } = req.body;
        if (!message) return res.status(400).json({ error: "Message is required" });

        console.log(`\n---> Received user message: "${message}"`);
        
        // Inject the financial data invisibly into the AI's prompt
        const enrichedPrompt = `Here is the user's current financial data: ${JSON.stringify(mockUserData)}\n\nUser's question: ${message}`;
        
        const result = await model.generateContent(enrichedPrompt);
        const reply = result.response.text();
        
        console.log("---> AI successfully analyzed data and replied!");
        res.json({ reply: reply });
    } catch (error) {
        console.error("\n*** GEMINI API ERROR ***");
        console.error(error);
        res.status(500).json({ error: "Failed to generate response" });
    }
});

const PORT = 8000;
const server = app.listen(PORT, () => {
    console.log(`\n✅ FinTech Backend successfully running on http://localhost:${PORT}`);
    console.log(`⏳ Waiting for messages from the app...\n`);
});

server.on('error', (error) => {
    console.error('\n❌ SERVER PORT ERROR:', error.message);
});

setInterval(() => {}, 60000);