require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI, SchemaType } = require('@google/generative-ai');

const app = express();
app.use(cors());
app.use(express.json({ limit: '20mb' }));

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// ==========================================
// 1. ROUTER AGENT (Fast Intent Classification)
// ==========================================
const routerModel = genAI.getGenerativeModel({ 
    model: "gemini-3.5-flash-lite", 
    systemInstruction: `Classify the user's financial query into exactly one of these strings: 
    "expense" (spending, budgets, general questions), 
    "bills" (subscriptions, recurring costs, due dates), 
    "investments" (stocks, crypto, portfolio growth).
    Return ONLY the single word. Do not include punctuation.`,
});

// ==========================================
// 2. WORKER AGENTS (Specialized Tasks & Schemas)
// ==========================================

const baseSchemaProperties = {
    ai_reply: { type: SchemaType.STRING, description: "Your conversational financial advice." },
    total_balance: { type: SchemaType.NUMBER, description: "The total combined balance." }
};

// A. Expense & Analyst Agent
const expenseModel = genAI.getGenerativeModel({
    model: "gemini-3.5-flash-lite",
    systemInstruction: "You are the Expense & Analyst Agent. Analyze the user's spending habits against their budget and provide actionable advice.",
    generationConfig: {
        responseMimeType: "application/json",
        responseSchema: {
            type: SchemaType.OBJECT,
            properties: {
                ...baseSchemaProperties,
                budget_status: { type: SchemaType.STRING, description: "'On Track', 'Nearing Limit', or 'Over Budget'" },
                category_breakdown: {
                    type: SchemaType.ARRAY,
                    items: {
                        type: SchemaType.OBJECT,
                        properties: { category: { type: SchemaType.STRING }, amount: { type: SchemaType.NUMBER } }
                    }
                }
            },
            required: ["ai_reply", "total_balance", "budget_status", "category_breakdown"]
        }
    }
});

// B. Bills & Subscriptions Agent
const billsModel = genAI.getGenerativeModel({
    model: "gemini-3.5-flash-lite",
    systemInstruction: "You are the Bill & Subscription Agent. Scan the data for fixed overhead costs, alert the user before renewal dates, and flag duplicate subscriptions.",
    generationConfig: {
        responseMimeType: "application/json",
        responseSchema: {
            type: SchemaType.OBJECT,
            properties: {
                ...baseSchemaProperties,
                upcoming_bills: {
                    type: SchemaType.ARRAY,
                    items: {
                        type: SchemaType.OBJECT,
                        properties: { bill_name: { type: SchemaType.STRING }, due_date: { type: SchemaType.STRING }, amount: { type: SchemaType.NUMBER } }
                    }
                }
            },
            required: ["ai_reply", "total_balance", "upcoming_bills"]
        }
    }
});

// C. Investment & Portfolio Agent
const investModel = genAI.getGenerativeModel({
    model: "gemini-3.5-flash-lite",
    systemInstruction: "You are the Investment Agent. Evaluate asset allocation, net asset growth, and portfolio performance.",
    generationConfig: {
        responseMimeType: "application/json",
        responseSchema: {
            type: SchemaType.OBJECT,
            properties: {
                ...baseSchemaProperties,
                portfolio_health: { type: SchemaType.STRING, description: "Overall rating of the portfolio." },
                category_breakdown: {
                    type: SchemaType.ARRAY,
                    items: {
                        type: SchemaType.OBJECT,
                        properties: { category: { type: SchemaType.STRING }, amount: { type: SchemaType.NUMBER } }
                    }
                }
            },
            required: ["ai_reply", "total_balance", "portfolio_health", "category_breakdown"]
        }
    }
});

// D. Receipt OCR Agent
const receiptModel = genAI.getGenerativeModel({
    model: "gemini-3.5-flash-lite",
    systemInstruction: "You are a Receipt Data Extraction Agent. Read the provided image and extract the merchant name, date, total amount, and suggest an expense category.",
    generationConfig: {
        responseMimeType: "application/json",
        responseSchema: {
            type: SchemaType.OBJECT,
            properties: {
                merchant: { type: SchemaType.STRING, description: "Name of the store or business" },
                date: { type: SchemaType.STRING, description: "Date on the receipt (YYYY-MM-DD)" },
                total: { type: SchemaType.NUMBER, description: "Final total amount charged" },
                category: { type: SchemaType.STRING, description: "Suggested budget category (e.g., Groceries, Transport, Dining)" }
            },
            required: ["merchant", "date", "total", "category"]
        }
    }
});

// ==========================================
// 3. MOCK DATABASE
// ==========================================
const mockUserData = {
    accounts: { checking: 2450.00, savings: 8500.00, crypto: 1200.00, stocks: 4500.00 },
    monthlyBudget: 300.00,
    recentTransactions: [
        { date: "2026-10-14", merchant: "Whole Foods", amount: 120.50, category: "Groceries" },
        { date: "2026-10-16", merchant: "Netflix", amount: 15.99, category: "Subscriptions", recurring: true }
    ],
    upcomingBills: [
        { name: "Electricity", dueDate: "2026-10-20", amount: 85.00 },
        { name: "Spotify", dueDate: "2026-10-22", amount: 10.99 }
    ]
};

// ==========================================
// 4. AUTHENTICATION ROUTE
// ==========================================
app.post('/login', (req, res) => {
    const { username, password } = req.body;
    console.log(`\n---> Login attempt for user: "${username}"`);

    if (username === 'souvik' && password === 'password123') {
        res.json({ access_token: "mock-jwt-token-xyz789" });
    } else {
        res.status(401).json({ error: "Invalid username or password" });
    }
});

// ==========================================
// 5. MAIN API ROUTE (The Orchestrator)
// ==========================================
app.post('/api/chat', async (req, res) => {
    try {
        const { message } = req.body;
        if (!message) return res.status(400).json({ error: "Message is required" });

        console.log(`\n---> 1. Received user message: "${message}"`);
        
        const routeResult = await routerModel.generateContent(`User message: "${message}"`);
        const intent = routeResult.response.text().trim().toLowerCase();
        console.log(`---> 2. Router classified intent as: [${intent}]`);

        let selectedModel;
        switch (intent) {
            case 'bills':
                selectedModel = billsModel;
                break;
            case 'investments':
                selectedModel = investModel;
                break;
            case 'expense':
            default:
                selectedModel = expenseModel; 
                break;
        }

        const enrichedPrompt = `User's financial data: ${JSON.stringify(mockUserData)}\n\nUser's question: ${message}`;
        const result = await selectedModel.generateContent(enrichedPrompt);
        
        const aiData = JSON.parse(result.response.text());
        console.log(`---> 3. Worker Agent generated structured response.`);
        res.json(aiData);

    } catch (error) {
        console.error("\n*** API ERROR ***\n", error);
        res.status(500).json({ error: "Failed to process the multi-agent request." });
    }
});

// ==========================================
// 6. RECEIPT OCR ROUTE (Vision Integration)
// ==========================================
app.post('/api/receipt', async (req, res) => {
    try {
        const { base64Image, mimeType } = req.body;
        if (!base64Image) return res.status(400).json({ error: "No image provided" });

        console.log("\n---> [OCR Agent] Received receipt image for processing...");

        const imagePart = {
            inlineData: {
                data: base64Image,
                mimeType: mimeType || "image/jpeg"
            }
        };

        const prompt = "Extract the data from this receipt perfectly.";
        const result = await receiptModel.generateContent([prompt, imagePart]);
        
        const extractedData = JSON.parse(result.response.text());
        console.log("---> [OCR Agent] Extraction successful:", extractedData);
        
        res.json(extractedData);

    } catch (error) {
        console.error("\n*** OCR ERROR ***\n", error);
        res.status(500).json({ error: "Failed to read the receipt image." });
    }
});

const PORT = 8000;
app.listen(PORT, () => {
    console.log(`\n✅ Multi-Agent Backend successfully running on http://localhost:${PORT}`);
});