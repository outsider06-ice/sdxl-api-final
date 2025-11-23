import os
import time
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from diffusers import DiffusionPipeline
import io
import base64

# ==================== INITIALISATION ====================
app = Flask(__name__)
CORS(app)

print("=" * 50)
print("🚀 DÉMARRAGE API - MODÈLE LÉGER")
print("=" * 50)

# ==================== MODÈLE LÉGER ====================
print("📦 Étape 1/3: Chargement du modèle léger...")

try:
    # Utiliser Stable Diffusion 1.5 au lieu de SDXL (BEAUCOUP plus léger)
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # Optimisations mémoire
    pipe.enable_attention_slicing()
    pipe.enable_memory_efficient_attention()
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        device = "cuda"
        print("✅ Modèle chargé sur GPU!")
    else:
        pipe = pipe.to("cpu") 
        device = "cpu"
        print("✅ Modèle chargé sur CPU!")
        
except Exception as e:
    print(f"❌ ERREUR: {e}")
    exit(1)

# ==================== STYLES ====================
STYLES = {
    "realistic": "photorealistic, realistic lighting, professional photography",
    "cinematic": "cinematic, movie still, dramatic lighting",
    "horror": "horror atmosphere, scary, dark fantasy, eerie lighting",
    "fantasy": "fantasy art, magical realm, mythical creatures",
    "surrealiste": "surrealism, dreamlike, bizarre, impossible reality",
    "cartoon": "cartoon style, bold outlines, vibrant colors",
    "pixart": "pixel art, 8-bit, retro gaming",
    "abstract": "abstract art, geometric shapes, expressive"
}

print("🎨 Étape 2/3: Styles chargés - 8 styles disponibles")

# ==================== ROUTES API ====================
@app.route('/')
def home():
    return jsonify({
        "service": "Stable Diffusion API - Render",
        "status": "active",
        "device": device,
        "styles_available": list(STYLES.keys())
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "device": device})

@app.route('/styles')
def get_styles():
    return jsonify({
        "available_styles": list(STYLES.keys()),
        "total_styles": len(STYLES)
    })

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get('prompt', 'a beautiful landscape')
        style = data.get('style', 'cinematic')
        
        # Réduire la résolution pour économiser mémoire
        width = min(data.get('width', 384), 512)
        height = min(data.get('height', 384), 512)
        steps = min(data.get('steps', 15), 20)
        
        print(f"🎨 Génération - Style: {style}")
        
        style_prompt = STYLES.get(style, STYLES['cinematic'])
        full_prompt = f"{prompt}, {style_prompt}, masterpiece, best quality"
        
        print("🔄 Génération en cours...")
        start_time = time.time()
        
        image = pipe(
            prompt=full_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=7.5
        ).images[0]
        
        generation_time = time.time() - start_time
        print(f"✅ Image générée en {generation_time:.1f}s!")
        
        # Conversion base64
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode()
        
        return jsonify({
            "status": "success",
            "style": style,
            "image_data": f"data:image/png;base64,{img_base64}",
            "dimensions": f"{width}x{height}",
            "generation_time": f"{generation_time:.1f}s",
            "device": device
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🌍 Démarrage sur le port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
