require('dotenv').config();

async function checkModels() {
    console.log("Checking Google's servers for your available models...\n");
    const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${process.env.GEMINI_API_KEY}`;
    
    try {
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.error) {
            console.error("API Error:", data.error.message);
            return;
        }

        console.log("✅ AVAILABLE MODELS FOR YOUR ACCOUNT:");
        const validModels = data.models.filter(m => 
            m.supportedGenerationMethods && m.supportedGenerationMethods.includes("generateContent")
        );
        
        validModels.forEach(m => {
            console.log(`- ${m.name.replace('models/', '')}`);
        });
        
    } catch (error) {
        console.error("Network Error:", error);
    }
}

checkModels();