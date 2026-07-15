import { ChatOpenAI } from "@langchain/openai";
import type { UsageMetadata } from "@langchain/core/messages";

export interface SpriteResult {
  csv: string;
  palette: Record<string, string>;
  usage: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
    costUSD: string;
  };
}

// DeepSeek V4 Flash pricing per 1M tokens (as of 2025-07)
const INPUT_PRICE_PER_1M = 0.14;
const OUTPUT_PRICE_PER_1M = 0.28;

const SYSTEM_PROMPT = `You are a pixel art generator. You output ONLY valid JSON — no markdown, no explanation, no code fences. Your entire response must parse as JSON.

The JSON object has exactly two keys:

1. "csv" — a string containing EXACTLY 64 rows, each row containing EXACTLY 64 comma-separated integers from 0 to 9. Rows are joined by newline characters (\\n). No trailing newline. No spaces. NO MORE THAN 64 ROWS. NO LESS THAN 64 ROWS. Count them: 64 rows, 64 values per row.
   - 0 means transparent/background/empty space. Use it generously for empty areas around the subject.
   - Values 1-9 represent different colors (defined in the palette).
   - The sprite must be a recognizable representation of the user's description, centered in the 64x64 grid.
   - Use solid blocks of color (no dithering, no gradients). Think retro 8-bit pixel art.
   - The subject should occupy roughly 30-60% of the grid area, centered.

2. "palette" — an object with keys "0" through "9" mapping to CSS color strings (hex like "#FF8844" or rgba like "rgba(255,136,68,1)").
   - Key "0" MUST be a fully transparent color: "rgba(0,0,0,0)".
   - Keys "1" through "9" should be the colors used in the CSV, chosen to match the described subject.
   - Use distinct, vibrant colors so each index is clearly visible.
   - Every key "0" through "9" MUST be present exactly once.

CRITICAL CONSTRAINTS — VIOLATING ANY OF THESE IS AN ERROR:
- The CSV must have EXACTLY 64 rows. Count them before outputting.
- Each row must have EXACTLY 64 comma-separated values. No more, no less.
- Use ONLY the digits 0-9 in the CSV. No other characters between commas.
- All 10 palette keys ("0" through "9") MUST be present.
- Output ONLY the JSON object. No markdown wrapping, no explanation text.`;

function buildUserPrompt(description: string): string {
  return `Generate a 64x64 pixel art sprite of: ${description}`;
}

function sanitizeCsv(csv: string): string {
  const rows = csv.trim().split("\n");

  // Trim or pad to exactly 64 rows
  let normalized: string[];
  if (rows.length > 64) {
    normalized = rows.slice(0, 64);
  } else if (rows.length < 64) {
    const emptyRow = Array(64).fill("0").join(",");
    normalized = [...rows];
    while (normalized.length < 64) {
      normalized.push(emptyRow);
    }
  } else {
    normalized = rows;
  }

  // Ensure each row has exactly 64 columns
  return normalized
    .map((row) => {
      const cols = row.split(",");
      if (cols.length > 64) return cols.slice(0, 64).join(",");
      if (cols.length < 64) {
        return cols.concat(Array(64 - cols.length).fill("0")).join(",");
      }
      return row;
    })
    .join("\n");
}

function parseResponse(text: string): { csv: string; palette: Record<string, string> } {
  // Strip markdown code fences if the model ignores instructions
  let json = text.trim();
  if (json.startsWith("```")) {
    json = json.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "");
  }

  const parsed = JSON.parse(json);

  // Validate and sanitize csv
  if (typeof parsed.csv !== "string") {
    throw new Error("Response missing 'csv' field");
  }
  const csv = sanitizeCsv(parsed.csv);

  // Validate palette
  if (typeof parsed.palette !== "object" || parsed.palette === null) {
    throw new Error("Response missing 'palette' field");
  }
  const palette: Record<string, string> = {};
  for (let i = 0; i <= 9; i++) {
    const key = String(i);
    if (typeof parsed.palette[key] !== "string") {
      throw new Error(`Palette missing key "${key}"`);
    }
    palette[key] = parsed.palette[key];
  }

  return { csv, palette };
}

function computeCost(usage: UsageMetadata): string {
  const inputCost = (usage.input_tokens / 1_000_000) * INPUT_PRICE_PER_1M;
  const outputCost = (usage.output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M;
  const total = inputCost + outputCost;
  return total < 0.0001 ? "<$0.0001" : `$${total.toFixed(4)}`;
}

export async function generateSprite(
  description: string
): Promise<SpriteResult> {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    throw new Error("DEEPSEEK_API_KEY environment variable is not set");
  }

  const model = new ChatOpenAI({
    model: "deepseek-v4-flash",
    temperature: 0.7,
    maxTokens: 16384,
    configuration: {
      baseURL: "https://api.deepseek.com/v1",
      apiKey,
    },
  });

  const response = await model.invoke([
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: buildUserPrompt(description) },
  ]);

  const text =
    typeof response.content === "string"
      ? response.content
      : Array.isArray(response.content)
        ? response.content.map((c) => (typeof c === "string" ? c : JSON.stringify(c))).join("")
        : "";

  const { csv, palette } = parseResponse(text);

  const usage = response.usage_metadata;
  if (!usage) {
    throw new Error("No token usage metadata in response");
  }

  return {
    csv,
    palette,
    usage: {
      inputTokens: usage.input_tokens,
      outputTokens: usage.output_tokens,
      totalTokens: usage.total_tokens,
      costUSD: computeCost(usage),
    },
  };
}
