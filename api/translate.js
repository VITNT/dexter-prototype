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
        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key=${API_KEY}`;

        const prompt = `You are an expert ASL translator. Convert this raw ASL gloss string into a natural, grammatically correct English sentence.


        The input contains merged words with no spaces. First, split the text into words.
        When splitting:
        - Prefer word groupings that form a coherent, meaningful sentence.
        - Avoid splits that create unrelated or nonsensical words.
        - Use context to choose the most logical interpretation.


        If multiple splits are possible, choose the one that makes the most sense semantically.
        Then translate into a clear, natural English sentence.

        NO THINKING: Do not show any internal reasoning, chain-of-thought, or "Option 1 / Option 2" analysis.

        NO STEPS: Do not list the word-splitting process.

        ONLY THE RESULT: Output ONLY the final English sentence and nothing else.

        Output exactly one short, concise sentence.
        Do not include explanations, intermediate steps, or alternative interpretations.
       
        Raw ASL: "${aslString}"`;


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


        const translation = data.candidates[0].content.parts[0].text;


        return res.status(200).json({ translation: translation });


    } catch (error) {
        console.error("Server Crash:", error);
        return res.status(500).json({ error: "Failed to connect to the AI model." });
    }
}