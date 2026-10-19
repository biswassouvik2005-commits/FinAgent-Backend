require('dotenv').config();
const fs = require('fs');
const path = require('path');
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
    "investments" (stocks, crypto, portfolio growth),
    "log_expense" (when the user states or speaks an expense, purchase, or money spent to record, OR pastes an automated bank/card transaction SMS alert, email receipt, or payment notification, e.g. "Chase Alert: You made a $24.50 transaction at TRADER JOE'S #542 on Oct 19 with card ending 4321. Avail bal: $2,425.50", "Citi Alert: A charge of $68.12 was approved at SHELL OIL 574423 on 10/19/2026. Ref: TXN987261. Balance: $1,280.00", "Spent $14 on lunch at Subway").
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

// E. Conversational & Bank Alert Expense Logging Agent (4th Route Worker)
const logExpenseModel = genAI.getGenerativeModel({
    model: "gemini-3.5-flash-lite",
    systemInstruction: `You are the Conversational & Bank Alert Expense Logging Agent.
When a user types or speaks an expense OR pastes an automated transaction SMS alert (e.g. Chase, Citi, Wells Fargo, Amex, Apple Card, Venmo, PayPal) or email receipt:
1. amount: The exact dollar amount spent/charged as a positive number (e.g. 24.50).
2. merchant: Clean store or business name (strip terminal numbers, store hashes, or POS codes like "#542", "574423", e.g. "Trader Joe's", "Shell Oil").
3. category: Suggested budget category (e.g. Dining, Groceries, Transportation, Shopping, Entertainment, Utilities, Healthcare, Other).
4. date: Transaction date formatted as YYYY-MM-DD or readable string, or "Today".
5. transaction_id: Reference number, transaction ID, confirmation code, or auth code if present (or null/empty if none).
6. remaining_balance: The account balance or available balance reported in the alert as a number (or null if not mentioned).
7. card_or_account: The card or account mentioned in the alert (e.g. "Card ...4321", "Checking ...1234", "Apple Card") or null.
8. is_bank_alert: true if this is an automated bank alert/SMS/push/email, false if a casual statement.
9. ai_reply: A concise, friendly confirmation message confirming the expense has been recorded and if the balance was reconciled.`,
    generationConfig: {
        responseMimeType: "application/json",
        responseSchema: {
            type: SchemaType.OBJECT,
            properties: {
                ai_reply: { type: SchemaType.STRING, description: "Friendly confirmation message." },
                amount: { type: SchemaType.NUMBER, description: "The amount spent in USD as a positive number." },
                merchant: { type: SchemaType.STRING, description: "Clean merchant or business name." },
                category: { type: SchemaType.STRING, description: "Budget category (e.g. Dining, Groceries, Transportation)." },
                date: { type: SchemaType.STRING, description: "Transaction date or 'Today'." },
                transaction_id: { type: SchemaType.STRING, description: "Transaction ID or reference number if found." },
                remaining_balance: { type: SchemaType.NUMBER, description: "Available/remaining balance if specified." },
                card_or_account: { type: SchemaType.STRING, description: "Card or account used if mentioned." },
                is_bank_alert: { type: SchemaType.BOOLEAN, description: "Whether this came from an automated bank SMS/alert." }
            },
            required: ["ai_reply", "amount", "merchant", "category"]
        }
    }
});

// Robust NLP fallback parser for zero-failure local/offline expense and bank SMS alert extraction
function fallbackParseExpense(text) {
    if (!text || typeof text !== 'string') {
        return {
            amount: 0,
            merchant: 'Expense',
            category: 'General',
            date: 'Today',
            is_bank_alert: false,
            ai_reply: 'Could not parse expense.'
        };
    }
    const clean = text.trim();
    const lower = clean.toLowerCase();

    // Detect bank alert indicators
    const isBankAlert = /(?:alert|notification|authorized|approved|declined|charge of|transaction of|purchase of|avail(?:able)?\s*bal|ending bal|acct bal|balance:|ref:|txn|ref\s*#|card ending|ending in \d{4})/i.test(clean) ||
        /^(?:chase|citi|wells fargo|bofa|bank of america|amex|apple card|capital one|venmo|paypal|discover)/i.test(clean);

    // 1. Extract remaining/available balance (e.g. "Avail bal: $2,425.50", "Balance: $1,280.00", "Available balance: $3,120.45")
    let remaining_balance;
    const balanceMatch = clean.match(/(?:avail(?:able)?\s*bal(?:ance)?|total\s*bal(?:ance)?|acct\s*bal(?:ance)?|ending\s*bal(?:ance)?|bal(?:ance)?):?\s*\$?\s*([\d,]+(?:\.\d{2})?)/i);
    if (balanceMatch && balanceMatch[1]) {
        const parsedBal = parseFloat(balanceMatch[1].replace(/,/g, ''));
        if (Number.isFinite(parsedBal)) {
            remaining_balance = parsedBal;
        }
    }

    // 2. Extract transaction amount (e.g. "$24.50 transaction", "charge of $68.12", "$14 on lunch")
    let amount = 0;
    const alertAmtMatch = clean.match(/(?:charge of|transaction of|purchase of|made a|payment of|spent|paid|amount:?)\s*\$?\s*([\d,]+(?:\.\d{1,2})?)/i);
    if (alertAmtMatch && alertAmtMatch[1]) {
        amount = parseFloat(alertAmtMatch[1].replace(/,/g, ''));
    } else {
        const allDollarMatches = [...clean.matchAll(/\$\s*([\d,]+(?:\.\d{1,2})?)/gi)];
        for (const m of allDollarMatches) {
            const val = parseFloat(m[1].replace(/,/g, ''));
            if (remaining_balance != null && Math.abs(val - remaining_balance) < 0.001) {
                continue;
            }
            amount = val;
            break;
        }
        if (!amount) {
            const genericNumber = clean.match(/(?:^|\s)(\d+(?:\.\d{1,2})?)(?:\s*(?:dollars?|bucks?|usd))?/i);
            if (genericNumber) {
                amount = parseFloat(genericNumber[1]);
            }
        }
    }

    // 3. Extract transaction ID / reference code (e.g. "Ref: TXN987261", "Txn ID: 59382", "Ref #12345")
    let transaction_id;
    const txnMatch = clean.match(/(?:ref(?:erence)?\s*(?:#|id|no\.?)?|txn(?:\s*id)?|auth(?:\s*code)?|confirmation\s*#?)\s*[:#]?\s*([A-Za-z0-9-]+)/i);
    if (txnMatch && txnMatch[1]) {
        transaction_id = txnMatch[1].trim();
    }

    // 4. Extract card or account (e.g. "card ending 4321", "card ending in 8899", "Apple Card", "Venmo", "Checking ...1234")
    let card_or_account;
    const cardEndingMatch = clean.match(/(?:card|acct|account)\s*(?:ending(?:\s*in)?|#)?\s*[:#]?\s*(\d{4})/i);
    if (cardEndingMatch && cardEndingMatch[1]) {
        card_or_account = `Card ending ${cardEndingMatch[1].trim()}`;
    } else if (/apple card/i.test(clean)) {
        card_or_account = 'Apple Card';
    } else if (/venmo/i.test(clean)) {
        card_or_account = 'Venmo';
    } else if (/chase/i.test(clean)) {
        card_or_account = 'Chase Checking';
    } else if (/citi/i.test(clean)) {
        card_or_account = 'Citi Card';
    }

    // 5. Extract merchant (e.g. "at TRADER JOE'S #542", "at SHELL OIL 574423", "Paid John Doe $35", "at Subway")
    let merchant = '';
    const atMatch = clean.match(/(?:at|from|to|with|paid)\s+([A-Za-z0-9\s'&.#\-]+?)(?:\s*(?:\.|\$|\s+(?:for|on|yesterday|today|with|using|in|ref|ref:|txn|bal|balance|avail|total)|$))/i);
    if (atMatch && atMatch[1]) {
        merchant = atMatch[1].trim();
    }
    if (!merchant || merchant.toLowerCase() === 'card' || merchant.toLowerCase() === 'you') {
        const commonMerchants = ['subway', 'starbucks', 'walmart', 'target', 'amazon', 'uber', 'lyft', 'shell', 'mcdonalds', 'whole foods', "trader joe's"];
        const found = commonMerchants.find(m => lower.includes(m));
        merchant = found ? (found.charAt(0).toUpperCase() + found.slice(1)) : (isBankAlert ? 'Card Transaction' : 'Expense');
    }

    // Clean store #, POS, terminal digits
    merchant = merchant
        .replace(/\s*(?:#\d+|\b\d{3,}\b|\bstore\b\s*\d*|\bpos\b).*$/i, '')
        .trim();
    merchant = merchant.replace(/[.#\-,]+$/, '').trim();

    if (merchant.length > 2 && merchant === merchant.toUpperCase()) {
        merchant = merchant.toLowerCase().split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    }

    // 6. Extract date (e.g. "on Oct 19", "on 10/19/2026", "on 10/18")
    let date = 'Today';
    const dateMatch = clean.match(/(?:on|date:?)\s+([A-Za-z]{3}\s+\d{1,2}(?:,\s*\d{4})?|\d{1,2}\/\d{1,2}(?:\/\d{2,4})?)/i);
    if (dateMatch && dateMatch[1]) {
        date = dateMatch[1].trim();
    }

    // 7. Extract category
    let category = 'General';
    const mLower = merchant.toLowerCase();
    if (/groceries|grocery|supermarket|apples|milk|produce|trader joe|whole foods/i.test(lower) || /trader joe|safeway|kroger|aldi|publix/i.test(mLower)) {
        category = 'Groceries';
    } else if (/lunch|dinner|breakfast|meal|food|subway|starbucks|mcdonalds|burger|sandwich|pizza|coffee|cafe|restaurant|dining|snack|drink/i.test(lower) || /starbucks|subway|mcdonald/i.test(mLower)) {
        category = 'Dining';
    } else if (/gas|fuel|shell|chevron|exxon|uber|lyft|taxi|subway fare|train|bus|transport|parking|toll/i.test(lower) || /shell|chevron|exxon|bp|mobil/i.test(mLower)) {
        category = 'Transportation';
    } else if (/movie|cinema|netflix|spotify|game|entertainment|concert|ticket/i.test(lower)) {
        category = 'Entertainment';
    } else if (/clothes|shoes|amazon|target|shopping|mall|electronics|store|retail/i.test(lower) || /target|walmart|amazon/i.test(mLower)) {
        category = 'Shopping';
    } else if (/bill|electric|power|water|internet|utility|utilities|wifi|phone|rent/i.test(lower)) {
        category = 'Utilities';
    } else if (/gym|pharmacy|medicine|doctor|health|dentist/i.test(lower) || /cvs|walgreens/i.test(mLower)) {
        category = 'Healthcare';
    }

    // 8. Generate appropriate AI reply
    let ai_reply = '';
    if (isBankAlert) {
        if (remaining_balance != null) {
            ai_reply = `Bank alert parsed: Logged $${amount.toFixed(2)} at ${merchant} under ${category}. Checking balance reconciled to $${remaining_balance.toLocaleString('en-US', { minimumFractionDigits: 2 })}!`;
        } else {
            ai_reply = `Bank alert parsed: Logged $${amount.toFixed(2)} at ${merchant} under ${category}. Added to your transactions!`;
        }
    } else {
        ai_reply = `Logged $${amount.toFixed(2)} at ${merchant} for ${category}. Added to your transactions!`;
    }

    return {
        amount,
        merchant,
        category,
        date,
        transaction_id,
        remaining_balance,
        card_or_account,
        is_bank_alert: isBankAlert,
        ai_reply
    };
}

// ==========================================
// 3. MOCK DATABASE & PERSISTENT STORAGE
// ==========================================
const DATA_FILE = path.join(__dirname, 'user_data.json');

const DEFAULT_USER_DATA = {
    accounts: { checking: 2450.00, savings: 8500.00, crypto: 1200.00, stocks: 4500.00 },
    monthlyBudget: 300.00,
    recentTransactions: [
        { id: 1, date: "2026-10-14", merchant: "Whole Foods", amount: 120.50, category: "Groceries", type: "expense" },
        { id: 2, date: "2026-10-16", merchant: "Netflix", amount: 15.99, category: "Subscriptions", recurring: true, type: "expense" }
    ],
    upcomingBills: [
        { name: "Electricity", dueDate: "2026-10-20", amount: 85.00 },
        { name: "Spotify", dueDate: "2026-10-22", amount: 10.99 }
    ]
};

let mockUserData = JSON.parse(JSON.stringify(DEFAULT_USER_DATA));

function saveUserData() {
    try {
        fs.writeFileSync(DATA_FILE, JSON.stringify(mockUserData, null, 2), 'utf8');
        console.log(`[Storage] Saved live user data to ${DATA_FILE}`);
    } catch (err) {
        console.error('[Storage] Error saving to user_data.json:', err.message);
    }
}

function loadUserData() {
    try {
        if (fs.existsSync(DATA_FILE)) {
            const raw = fs.readFileSync(DATA_FILE, 'utf8');
            const parsed = JSON.parse(raw);
            if (parsed && typeof parsed === 'object') {
                mockUserData = {
                    ...DEFAULT_USER_DATA,
                    ...parsed,
                    accounts: {
                        ...DEFAULT_USER_DATA.accounts,
                        ...(parsed.accounts || {})
                    },
                    recentTransactions: Array.isArray(parsed.recentTransactions)
                        ? parsed.recentTransactions
                        : DEFAULT_USER_DATA.recentTransactions
                };
                console.log(`[Storage] Successfully loaded live user data from ${DATA_FILE}`);
                return;
            }
        }
    } catch (err) {
        console.warn('[Storage] Error loading user_data.json, using default data:', err.message);
    }
    saveUserData();
}

loadUserData();

function recordExpense({ amount, merchant, category, date, transaction_id, remaining_balance, card_or_account, is_bank_alert }) {
    const numAmount = Math.max(0, Number(amount) || 0);
    const validMerchant = String(merchant || 'Expense').trim();
    const validCategory = String(category || 'General').trim();
    const validDate = String(date || new Date().toISOString().split('T')[0]).trim();

    const newTx = {
        id: Date.now(),
        date: validDate,
        merchant: validMerchant,
        amount: numAmount,
        category: validCategory,
        transaction_id: transaction_id || undefined,
        remaining_balance: (typeof remaining_balance === 'number' && Number.isFinite(remaining_balance)) ? remaining_balance : undefined,
        card_or_account: card_or_account || undefined,
        is_bank_alert: Boolean(is_bank_alert),
        type: 'expense'
    };

    if (numAmount > 0 || (typeof remaining_balance === 'number' && remaining_balance >= 0)) {
        mockUserData.recentTransactions.unshift(newTx);

        // If alert provides exact remaining/available balance, automatically reconcile checking account
        if (typeof remaining_balance === 'number' && Number.isFinite(remaining_balance) && remaining_balance >= 0) {
            mockUserData.accounts.checking = Math.round(remaining_balance * 100) / 100;
            console.log(`[Reconciliation] Reconciled checking account to alert balance: $${mockUserData.accounts.checking.toFixed(2)}`);
        } else {
            mockUserData.accounts.checking = Math.max(0, Math.round(((mockUserData.accounts.checking || 0) - numAmount) * 100) / 100);
            console.log(`[Live State] Deducted $${numAmount.toFixed(2)} from checking. New checking: $${mockUserData.accounts.checking.toFixed(2)}`);
        }

        saveUserData();
    }

    return newTx;
}

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
// 4B. SAFE-TO-SPEND CALCULATION ENGINE
// ==========================================
function calculateSafeToSpend(userData) {
    const now = new Date();
    const totalDays = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate();
    const daysLeft = Math.max(1, totalDays - now.getDate() + 1);
    const budget = Number(userData.monthlyIncome || userData.monthlyBudget || 0);
    const transactions = userData.transactions || userData.recentTransactions || [];
    const spentSoFar = transactions
        .filter(t => t.type === 'expense' || (t.amount && t.type !== 'income'))
        .reduce((sum, t) => sum + (Number(t.amount) || 0), 0);
    const bills = userData.bills || userData.upcomingBills || [];
    const upcomingBills = bills
        .filter(b => !b.paid)
        .reduce((sum, b) => sum + (Number(b.amount) || 0), 0);
    const remainingPool = budget - spentSoFar - upcomingBills;
    const dailySafeToSpend = budget > 0 ? Math.round((remainingPool / daysLeft) * 100) / 100 : 0;
    const dailyBaseline = budget > 0 ? budget / totalDays : 50;

    let status = 'green';
    let statusLabel = 'Guilt-Free Spending';
    if (budget === 0) {
        status = 'amber';
        statusLabel = 'Budget Not Set';
    } else if (dailySafeToSpend <= 0) {
        status = 'red';
        statusLabel = 'Spending Deficit';
    } else if (dailySafeToSpend < Math.min(25, dailyBaseline * 0.35)) {
        status = 'amber';
        statusLabel = 'Pacing Tight';
    }

    return {
        dailySafeToSpend,
        daysLeft,
        totalDays,
        monthlyBudget: budget,
        spentSoFar,
        upcomingBills,
        remainingPool,
        status,
        statusLabel
    };
}

app.get('/api/safe-to-spend', (req, res) => {
    res.json(calculateSafeToSpend(mockUserData));
});

// ==========================================
// 5. MAIN API ROUTE (The Orchestrator)
// ==========================================
app.post('/api/chat', async (req, res) => {
    try {
        const { message } = req.body;
        if (!message) return res.status(400).json({ error: "Message is required" });

        console.log(`\n---> 1. Received user message: "${message}"`);
        
        let intent = 'expense';
        try {
            const routeResult = await routerModel.generateContent(`User message: "${message}"`);
            intent = routeResult.response.text().trim().toLowerCase().replace(/[^a-z_]/g, '');
        } catch (routerErr) {
            console.warn("[Router] Gemini route fallback:", routerErr.message);
        }

        // Rule-based check if intent is not log_expense or was default
        if (intent !== 'log_expense') {
            const cleanMsg = message.toLowerCase();
            if (
                /(?:alert|notification|authorized|approved|declined|charge of|card ending|avail(?:able)? bal|balance:|ref:|txn|ref #)/i.test(cleanMsg) ||
                /(spent|paid|bought|cost|charged)\s*(\$|\d)/i.test(cleanMsg) ||
                /\$\s*\d+.*(on|at|for)/i.test(cleanMsg) ||
                /(on lunch|on dinner|for coffee|at subway|at starbucks|at walmart|for groceries)/i.test(cleanMsg) ||
                /(?:chase|citi|wells fargo|amex|apple card|venmo|paypal).*(?:\$|\d)/i.test(cleanMsg)
            ) {
                intent = 'log_expense';
            }
        }
        console.log(`---> 2. Router classified intent as: [${intent}]`);

        let selectedModel;
        switch (intent) {
            case 'log_expense':
                selectedModel = logExpenseModel;
                break;
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

        if (intent === 'log_expense') {
            let extractedData;
            try {
                const result = await logExpenseModel.generateContent(`Extract transaction or bank alert from: "${message}"`);
                extractedData = JSON.parse(result.response.text());
            } catch (err) {
                console.warn("[LogExpense] Gemini extraction fallback:", err.message);
                extractedData = fallbackParseExpense(message);
            }

            const amount = Number(extractedData.amount) || 0;
            const merchant = extractedData.merchant || "Expense";
            const category = extractedData.category || "General";
            const date = extractedData.date || new Date().toISOString().split('T')[0];
            const transaction_id = extractedData.transaction_id || undefined;
            const remaining_balance = (typeof extractedData.remaining_balance === 'number' && Number.isFinite(extractedData.remaining_balance))
                ? extractedData.remaining_balance
                : undefined;
            const card_or_account = extractedData.card_or_account || undefined;
            const is_bank_alert = Boolean(extractedData.is_bank_alert);

            const newTx = recordExpense({
                amount,
                merchant,
                category,
                date,
                transaction_id,
                remaining_balance,
                card_or_account,
                is_bank_alert
            });

            const reply = extractedData.ai_reply || (remaining_balance != null
                ? `Bank alert parsed: Logged $${amount.toFixed(2)} at ${merchant} under ${category}. Checking balance reconciled to $${remaining_balance.toLocaleString('en-US', { minimumFractionDigits: 2 })}!`
                : `Logged $${amount.toFixed(2)} at ${merchant} under ${category}. Added to your transactions automatically!`);
            console.log(`---> 3. Log Expense Agent extracted:`, { amount, merchant, category, remaining_balance, transaction_id, is_bank_alert });

            const totalBalance = Math.round(Object.values(mockUserData.accounts).reduce((a, b) => a + b, 0) * 100) / 100;

            return res.json({
                ai_reply: reply,
                total_balance: totalBalance,
                accounts: mockUserData.accounts,
                logged_expense: {
                    amount,
                    merchant,
                    category,
                    date: newTx.date,
                    id: newTx.id,
                    transaction_id,
                    remaining_balance,
                    card_or_account,
                    is_bank_alert,
                    type: 'expense'
                },
                transaction: newTx,
                recentTransactions: mockUserData.recentTransactions,
                safe_to_spend: calculateSafeToSpend(mockUserData),
                intent: 'log_expense'
            });
        }

        const activeUserData = (req.body.userContext && Object.keys(req.body.userContext).length > 0)
            ? { ...mockUserData, ...req.body.userContext }
            : mockUserData;
        const safeToSpend = calculateSafeToSpend(activeUserData);

        const enrichedPrompt = `User's financial data: ${JSON.stringify(activeUserData)}\nDaily Safe-to-Spend: $${safeToSpend.dailySafeToSpend}/day (${safeToSpend.statusLabel}). Formula: ($${safeToSpend.monthlyBudget} budget - $${safeToSpend.spentSoFar} spent - $${safeToSpend.upcomingBills} bills) / ${safeToSpend.daysLeft} days remaining in month.\n\nUser's question: ${message}`;
        const result = await selectedModel.generateContent(enrichedPrompt);
        
        const aiData = JSON.parse(result.response.text());
        aiData.intent = intent;
        aiData.safe_to_spend = safeToSpend;
        console.log(`---> 3. Worker Agent generated structured response. Intent: [${intent}], Safe-to-spend: $${safeToSpend.dailySafeToSpend}/day`);
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

        const total = Number(extractedData.total) || 0;
        const merchant = extractedData.merchant || "Receipt Store";
        const category = extractedData.category || "General";
        const date = extractedData.date || new Date().toISOString().split('T')[0];

        // Push new item into mockUserData.recentTransactions and deduct total from mockUserData.accounts.checking
        const newTx = recordExpense({ amount: total, merchant, category, date });
        const totalBalance = Math.round(Object.values(mockUserData.accounts).reduce((a, b) => a + b, 0) * 100) / 100;

        res.json({
            merchant,
            date,
            total,
            category,
            transaction: newTx,
            accounts: mockUserData.accounts,
            total_balance: totalBalance,
            recentTransactions: mockUserData.recentTransactions
        });

    } catch (error) {
        console.error("\n*** OCR ERROR ***\n", error);
        res.status(500).json({ error: "Failed to read the receipt image." });
    }
});

// ==========================================
// 7. CONVERSATIONAL EXPENSE LOGGING ROUTE (Text & Voice)
// ==========================================
app.post('/api/log_expense', async (req, res) => {
    try {
        const { message, text } = req.body || {};
        const query = message || text;
        if (!query) return res.status(400).json({ error: "Message or text is required" });

        console.log(`\n---> [Log Expense Route] Processing: "${query}"`);
        let extractedData;
        try {
            const result = await logExpenseModel.generateContent(`Extract transaction from: "${query}"`);
            extractedData = JSON.parse(result.response.text());
        } catch (err) {
            extractedData = fallbackParseExpense(query);
        }

        const amount = Number(extractedData.amount) || 0;
        const merchant = extractedData.merchant || "Expense";
        const category = extractedData.category || "General";
        const date = extractedData.date || new Date().toISOString().split('T')[0];

        const newTx = recordExpense({ amount, merchant, category, date });
        const totalBalance = Math.round(Object.values(mockUserData.accounts).reduce((a, b) => a + b, 0) * 100) / 100;

        res.json({
            success: true,
            ai_reply: extractedData.ai_reply || `Logged $${amount.toFixed(2)} at ${merchant} under ${category}. Added to your transactions automatically!`,
            amount,
            merchant,
            category,
            transaction: newTx,
            accounts: mockUserData.accounts,
            logged_expense: {
                amount,
                merchant,
                category,
                date: newTx.date,
                id: newTx.id,
                type: 'expense'
            },
            total_balance: totalBalance,
            recentTransactions: mockUserData.recentTransactions
        });
    } catch (error) {
        console.error("\n*** LOG EXPENSE ERROR ***\n", error);
        res.status(500).json({ error: "Failed to process expense logging." });
    }
});

app.post('/log_expense', (req, res) => {
    req.url = '/api/log_expense';
    app.handle(req, res);
});

// ==========================================
// 8. USER DATA STATE ENDPOINTS (Live State & Charts)
// ==========================================
app.get('/api/user', (req, res) => {
    const totalBalance = Math.round(Object.values(mockUserData.accounts).reduce((a, b) => a + b, 0) * 100) / 100;
    res.json({
        ...mockUserData,
        total_balance: totalBalance
    });
});

app.get('/api/data', (req, res) => {
    const totalBalance = Math.round(Object.values(mockUserData.accounts).reduce((a, b) => a + b, 0) * 100) / 100;
    res.json({
        ...mockUserData,
        total_balance: totalBalance
    });
});

app.post('/api/user/reset', (req, res) => {
    mockUserData = JSON.parse(JSON.stringify(DEFAULT_USER_DATA));
    saveUserData();
    res.json({ success: true, message: "User data reset to default", data: mockUserData });
});

const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
if (require.main === module) {
    app.listen(PORT, () => {
        console.log(`\n✅ Multi-Agent Backend successfully running on http://localhost:${PORT}`);
    });
}

module.exports = {
    app,
    routerModel,
    logExpenseModel,
    expenseModel,
    billsModel,
    investModel,
    receiptModel,
    mockUserData,
    DEFAULT_USER_DATA,
    saveUserData,
    loadUserData,
    recordExpense,
    fallbackParseExpense
};