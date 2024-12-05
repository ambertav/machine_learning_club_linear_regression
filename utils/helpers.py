def calculate_bmi (weight, height) :
    if weight == 0 or height == 0 :
        return None
    weight_kg = weight * 0.45359237 
    height_m = height * 0.0254
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 1)