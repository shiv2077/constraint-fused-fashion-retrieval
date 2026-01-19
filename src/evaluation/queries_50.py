"""Generate 50 diverse evaluation queries for robust evaluation."""

# 20 Color-focused queries
COLOR_QUERIES = [
    "A person wearing a red shirt.",
    "Someone in a blue dress.",
    "A model in yellow clothing.",
    "Person wearing green jacket.",
    "Black outfit for formal event.",
    "White dress for evening wear.",
    "Orange top for casual day.",
    "Purple dress for party.",
    "Pink dress for summer.",
    "Brown leather jacket.",
    "Gray sweater for fall.",
    "Bright red lipstick with dress.",
    "Navy blue suit for office.",
    "Emerald green gown.",
    "Gold sequin dress.",
    "Silver metallic top.",
    "Maroon evening dress.",
    "Turquoise summer dress.",
    "Beige neutral outfit.",
    "Deep blue ocean dress.",
]

# 20 Garment-focused queries
GARMENT_QUERIES = [
    "Person wearing a leather jacket.",
    "Model in a cotton t-shirt.",
    "Someone in denim jeans.",
    "Silk blouse for professional.",
    "Wool sweater for winter.",
    "Linen dress for summer.",
    "Leather pants for night.",
    "Cotton shorts for casual.",
    "Formal business suit.",
    "Casual sneakers with outfit.",
    "High heels for formal.",
    "Flat shoes for comfort.",
    "Scarf and sweater combination.",
    "Hat with casual outfit.",
    "Gloves in winter wear.",
    "Cardigan over dress.",
    "Blazer with pants.",
    "Vest with shirt.",
    "Skirt and blouse.",
    "Jumpsuit for all day.",
]

# 20 Mixed/Complex queries
MIXED_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
    "Green dress for spring garden party.",
    "Black leather jacket with jeans for cool weather.",
    "White linen dress for beach vacation.",
    "Gold jewelry with evening gown.",
    "Denim jacket over floral dress.",
    "Wool coat for snowy weather.",
    "Striped shirt with khaki pants.",
    "Sequin top for night club.",
    "Vintage style dress for retro look.",
    "Sporty outfit with sneakers.",
    "Bohemian dress with accessories.",
    "Minimalist black outfit for modern look.",
    "Colorful patterned dress for fun.",
    "Silk pajamas for bedroom style.",
    "Business formal dress with heels.",
]

ALL_QUERIES = COLOR_QUERIES + GARMENT_QUERIES + MIXED_QUERIES

if __name__ == "__main__":
    print(f"Total queries: {len(ALL_QUERIES)}")
    print(f"\nColor-focused: {len(COLOR_QUERIES)}")
    print(f"Garment-focused: {len(GARMENT_QUERIES)}")
    print(f"Mixed/Complex: {len(MIXED_QUERIES)}")
    print(f"\nAll queries:")
    for i, q in enumerate(ALL_QUERIES, 1):
        print(f"{i:2d}. {q}")
