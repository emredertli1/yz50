def analyze_text_encoding(text: str):
    print(f"İncelenen Metin: \"{text}\"\n")
    print(f"{'Karakter':<10} | {'Unicode (Hex)':<15} | {'Unicode (Dec)':<15} | {'UTF-8 Byte (Hex)':<20} | {'Byte Sayısı':<12}")
    print("-" * 80)
    
    byte_counts = {}
    
    for char in text:
        # Unicode kod noktası (Integer ve Hexadecimal)
        code_point_dec = ord(char)
        code_point_hex = f"U+{code_point_dec:04X}"
        
        # UTF-8 byte temsili
        utf8_bytes = char.encode('utf-8')
        utf8_hex = " ".join([f"0x{b:02X}" for b in utf8_bytes])
        byte_len = len(utf8_bytes)
        
        # Görünmeyen boşluk karakterini net göstermek için
        display_char = "' '" if char == ' ' else char
        
        print(f"{display_char:<10} | {code_point_hex:<15} | {code_point_dec:<15} | {utf8_hex:<20} | {byte_len:<12}")
        
        # Karakter bazında byte sayısını gruplayalım
        if char not in byte_counts:
            byte_counts[char] = byte_len

    print("-" * 80)
    
    # 1 Byte ve 2 Byte tutan karakterlerin özeti
    one_byte_chars = sorted(list(set([c for c, b in byte_counts.items() if b == 1])))
    two_byte_chars = sorted(list(set([c for c, b in byte_counts.items() if b == 2])))
    
    print("\nÖZET:")
    print(f"1 Byte Tutan Karakterler (ASCII): {one_byte_chars}")
    print(f"2 Byte Tutan Karakterler (Genişletilmiş Latin/Türkçe): {two_byte_chars}")

# ç, ğ, ı, ö, ş, ü karakterlerinin tamamını içeren örnek bir cümle
cumle = "Fıstıkçı Şahap, ılık sütü ve poğaçayı özledi."
analyze_text_encoding(cumle)