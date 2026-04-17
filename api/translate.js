module.exports = async function handler(req, res) {


    if (req.method !== 'POST') {
        return res.status(405).json({ error: "Method Not Allowed" });
    }


    try {
        const aslString = req.body.aslString;


        if (!aslString) {
            return res.status(400).json({ error: "Missing ASL string" });
        }


        const API_KEY = process.env.GEMINI_API_KEY;
        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent?key=${API_KEY}`;

        const prompt = `You are an expert ASL translator. Translate the following raw ASL gloss string into a single natural, grammatically correct English sentence.
The input ASL gloss may contain merged words with no spaces. You must silently interpret the correct word boundaries based on context and semantics to form a coherent sentence before translating.

CRITICAL INSTRUCTION: You must respond ONLY with the final English sentence. Do not include any explanations, thinking process, markdown formatting, or bullet points.

Raw ASL: HELLOHOWAREYOU
Translation: Hello, how are you?

Raw ASL: IWORKSTORE
Translation: I work at the store.

Raw ASL: ${aslString}
Translation:`;


        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                contents: [{ parts: [{ text: prompt }] }]
            })
        });


        const data = await response.json();


        if (!response.ok) {
            console.error("API Error Response:", data);
            return res.status(response.status).json({ error: "Google API Error", details: data });
        }


        let translation = data.candidates[0].content.parts[0].text.trim();

        // Fallback: if the model still outputs thinking, attempt to clean it up
        translation = translation.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
        
        // Remove common prefixes it might mistakenly include
        if (translation.toLowerCase().startsWith('translation:')) {
            translation = translation.substring(12).trim();
        } else if (translation.toLowerCase().startsWith('* output:')) {
            translation = translation.substring(9).trim();
        }
        
        // Remove surrounding quotes if present
        if (translation.startsWith('"') && translation.endsWith('"')) {
            translation = translation.slice(1, -1).trim();
        }

        return res.status(200).json({ translation: translation });


    } catch (error) {
        console.error("Server Crash:", error);
        return res.status(500).json({ error: "Failed to connect to the AI model." });
    }
}