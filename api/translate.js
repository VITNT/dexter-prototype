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

        // 🔥 Short + strict prompt (better compliance)
        const prompt = `
Translate this ASL gloss into natural English.

Rules:
- Output ONE short sentence only
- No explanations
- No reasoning
- No steps
- No extra text

ASL: ${aslString}
`;

        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                contents: [{ parts: [{ text: prompt }] }],
                generationConfig: {
                    temperature: 0.2,
                    maxOutputTokens: 50,
                    topP: 0.8
                }
            })
        });

        const data = await response.json();

        if (!response.ok) {
            console.error("API Error Response:", data);
            return res.status(response.status).json({
                error: "Google API Error",
                details: data
            });
        }

        // ✅ Safe extraction (prevents crashes if structure changes)
        let rawText =
            data?.candidates?.[0]?.content?.parts?.[0]?.text || "";

        // 🔧 HARD CLEAN: remove reasoning / chain-of-thought
        let cleaned = rawText
            .replace(/[\s\S]*?→/g, "")        // remove "X → Y" reasoning
            .replace(/Wait,.*?\n/g, "")
            .replace(/Maybe.*?\n/g, "")
            .replace(/Let’s.*?\n/g, "")
            .replace(/If.*?\n/g, "")
            .replace(/First,.*?\n/g, "")
            .replace(/Then,.*?\n/g, "")
            .trim();

        // Keep only last line (usually final answer)
        cleaned = cleaned.split("\n").pop().trim();

        // Extra safety: remove quotes if model wraps answer
        cleaned = cleaned.replace(/^["']|["']$/g, "");

        if (!cleaned) {
            cleaned = "Unable to generate translation.";
        }

        return res.status(200).json({ translation: cleaned });

    } catch (error) {
        console.error("Server Crash:", error);
        return res.status(500).json({
            error: "Failed to connect to the AI model."
        });
    }
};