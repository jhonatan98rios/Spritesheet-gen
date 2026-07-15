import { NextResponse } from "next/server";
import { generateSprite } from "@/app/services/sprite-generator";

export async function POST(request: Request) {
  try {
    const { description } = await request.json();

    if (!description || typeof description !== "string" || description.trim().length === 0) {
      return NextResponse.json(
        { error: "Description is required" },
        { status: 400 }
      );
    }

    const result = await generateSprite(description.trim());
    return NextResponse.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("Sprite generation failed:", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
