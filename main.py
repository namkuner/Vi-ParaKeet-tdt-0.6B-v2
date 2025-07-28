# import nemo.collections.asr as nemo_asr
# asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
# print(asr_model)
# output = asr_model.transcribe(['/content/First Battle Audio.mp3'],timestamps=True)

import nemo.collections.asr as nemo_asr
import os
import wget
# Load the model
# asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
#
# model_dir = "parakeet-tdt-0.6b-v2"
# os.makedirs(model_dir, exist_ok=True)
# model_path = os.path.join(model_dir, "parakeet-tdt-0.6b-v2.nemo")
# asr_model.save_to(model_path)
#
# import nemo.collections.asr as nemo_asr
# import os
# import wget
# # Load the model
# asr_model = nemo_asr.models.ASRModel.restore_from("experiments/Speech_To_Text_Finetuning_parakeet_v2_tdt_hindi/2025-07-24_07-40-54/checkpoints/Speech_To_Text_Finetuning_parakeet_v2_tdt_hindi.nemo")
# # print(asr_model)
# output = asr_model.transcribe(['audio_datasets/0/1.wav'], timestamps=True)
# print(output)
# =====================================================
# CÁCH 2: Hiển thị vocabulary (danh sách các token)
# =====================================================
# print("\n=== VOCABULARY ===")
# if hasattr(asr_model.tokenizer, 'vocab'):
#     vocab = asr_model.tokenizer.vocab
#     print(f"Number of tokens: {len(vocab)}")
#     print(vocab)
#
# # =====================================================
# # CÁCH 3: Hiển thị special tokens
# # =====================================================
# print("\n=== SPECIAL TOKENS ===")
# special_attrs = ['blank_token', 'unk_token', 'pad_token', 'bos_token', 'eos_token']
# for attr in special_attrs:
#     if hasattr(asr_model.tokenizer, attr):
#         token = getattr(asr_model.tokenizer, attr)
#         print(f"{attr}: '{token}'")
#
# # =====================================================
# # CÁCH 4: Hiển thị tokenizer config
# # =====================================================
# print("\n=== TOKENIZER CONFIG ===")
# if hasattr(asr_model.tokenizer, 'cfg'):
#     print("Tokenizer configuration:")
#     print(asr_model.tokenizer.cfg)
#
# # =====================================================
# # CÁCH 5: Test tokenizer với text mẫu
# # =====================================================
# print("\n=== TEST TOKENIZER ===")
# test_text = "Hello world, this is a test"
# if hasattr(asr_model.tokenizer, 'text_to_tokens'):
#     tokens = asr_model.tokenizer.text_to_tokens(test_text)
#     print(f"Input text: '{test_text}'")
#     print(f"Tokens: {tokens}")
#
#     # Chuyển ngược lại
#     if hasattr(asr_model.tokenizer, 'tokens_to_text'):
#         reconstructed = asr_model.tokenizer.tokens_to_text(tokens)
#         print(f"Reconstructed: '{reconstructed}'")
# # =====================================================
# # CÁCH 6: Hiển thị token-to-id mapping
# # =====================================================
# print("\n=== TOKEN-TO-ID MAPPING ===")
# if hasattr(asr_model.tokenizer, 'tokens_to_ids'):
#     sample_tokens = ['a', 'the', 'hello', '<unk>']
#     for token in sample_tokens:
#         try:
#             token_id = asr_model.tokenizer.tokens_to_ids([token])
#             print(f"Token '{token}' -> ID: {token_id}")
#         except:
#             print(f"Token '{token}' -> Not found")
#
# # =====================================================
# # CÁCH 7: Hiển thị chi tiết về tokenizer attributes
# # =====================================================
# print("\n=== CHI TIẾT TOKENIZER ATTRIBUTES ===")
# tokenizer_attrs = dir(asr_model.tokenizer)
# important_attrs = [attr for attr in tokenizer_attrs
#                    if not attr.startswith('_') and
#                    attr in ['vocab_size', 'blank_id', 'unk_id', 'pad_id', 'bos_id', 'eos_id']]
#
# for attr in important_attrs:
#     try:
#         value = getattr(asr_model.tokenizer, attr)
#         print(f"{attr}: {value}")
#     except:
#         print(f"{attr}: Error accessing")
#
# # =====================================================
# # CÁCH 8: So sánh với character-based tokenizer
# # =====================================================
# print("\n=== THÔNG TIN LOẠI TOKENIZER ===")
# tokenizer_type = type(asr_model.tokenizer).__name__
# print(f"Tokenizer class: {tokenizer_type}")
#
# # Kiểm tra xem là character-based hay subword-based
# if 'Char' in tokenizer_type:
#     print("This is a character-based tokenizer")
# elif 'SentencePiece' in tokenizer_type or 'BPE' in tokenizer_type:
#     print("This is a subword-based tokenizer")
# elif 'Word' in tokenizer_type:
#     print("This is a word-based tokenizer")
# else:
#     print(f"Tokenizer type: {tokenizer_type}")
#
#
# # =====================================================
# # CÁCH 9: Lưu tokenizer info vào file
# # =====================================================
# def save_tokenizer_info(tokenizer, filename="tokenizer_info.txt"):
#     """Lưu thông tin tokenizer vào file"""
#     with open(filename, 'w', encoding='utf-8') as f:
#         f.write(f"Tokenizer Type: {type(tokenizer).__name__}\n")
#         f.write(f"Vocab Size: {tokenizer.vocab_size}\n")
#         f.write("\n=== VOCABULARY ===\n")
#
#         if hasattr(tokenizer, 'vocab'):
#             for token, idx in tokenizer.vocab.items():
#                 f.write(f"{idx:4d}: {token}\n")
#
#
# print(f"\n=== SAVE INFO ===")
# save_tokenizer_info(asr_model.tokenizer)
# print("Tokenizer info saved to 'tokenizer_info.txt'")
#
# # =====================================================
# # CÁCH 10: Tìm hiểu về model architecture liên quan
# # =====================================================
# print("\n=== MODEL ARCHITECTURE INFO ===")
# print(f"Model type: {type(asr_model).__name__}")
# if hasattr(asr_model, 'decoder'):
#     print(f"Decoder vocab size: {asr_model.decoder.vocab_size if hasattr(asr_model.decoder, 'vocab_size') else 'N/A'}")
#
# # Hiển thị model config
# print("\n=== MODEL CONFIG (Tokenizer section) ===")
# if hasattr(asr_model, 'cfg') and 'tokenizer' in asr_model.cfg:
#     print("Tokenizer config from model:")
#     print(asr_model.cfg.tokenizer)
import torch
if __name__=="__main__":
    state = torch.load("experiments/Speech_To_Text_Finetuning_parakeet_v2_tdt_hindi/2025-07-24_07-40-54/checkpoints/Speech_To_Text_Finetuning_parakeet_v2_tdt_hindi--val_wer=8.6735-epoch=0-last.ckpt", map_location="cpu")
    print(state.keys())
    print(state["lr_schedulers"])
    print(state["optimizer_states"])