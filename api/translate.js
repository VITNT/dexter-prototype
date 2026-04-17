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

        // 🔥 Clean + strict prompt
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

        // ✅ Safe extraction
        let rawText =
            data?.candidates?.[0]?.content?.parts?.[0]?.text || "";

        // 🔧 FULL CLEANING PIPELINE
        let cleaned = rawText
            // Remove reasoning
            .replace(/[\s\S]*?→/g, "")
            .replace(/Wait,.*?\n/g, "")
            .replace(/Maybe.*?\n/g, "")
            .replace(/Let’s.*?\n/g, "")
            .replace(/If.*?\n/g, "")
            .replace(/First,.*?\n/g, "")
            .replace(/Then,.*?\n/g, "")

            // Remove markdown / bullets / numbering / leading junk
            .replace(/^[*\-•\d.\)\s"]+/gm, "")

            // Remove wrapping quotes
            .replace(/^["']|["']$/g, "")

            .trim();

        // Keep only last meaningful line
        cleaned = cleaned
            .split("\n")
            .filter(line => line.trim() !== "")
            .pop() || "";

        cleaned = cleaned.trim();

        // 🔧 Final formatting (guarantee sentence)
        if (cleaned) {
            cleaned =
                cleaned.charAt(0).toUpperCase() +
                cleaned.slice(1);

            if (!cleaned.endsWith(".")) {
                cleaned += ".";
            }
        } else {
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