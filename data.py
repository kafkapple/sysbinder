import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List, Optional, Dict
import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms


class GlobDataset(Dataset):
    def __init__(self, root, phase, img_size):
        self.root = root
        self.img_size = img_size
        self.total_imgs = sorted(glob.glob(root))

        if phase == 'train':
            self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
        elif phase == 'val':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
        elif phase == 'test':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
        else:
            pass

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        tensor_image = self.transform(image)
        return tensor_image

class TextGlobDataset(Dataset):
    def __init__(self, root: str, phase: str, tokenizer, max_length: int = 512):
        """
        텍스트 데이터를 처리하기 위한 Dataset 클래스
        
        Args:
            root: 텍스트 파일들이 있는 디렉토리 경로 (*.txt 파일들)
            phase: 'train', 'val', 'test' 중 하나
            tokenizer: 텍스트 토크나이저 (예: transformers의 토크나이저)
            max_length: 최대 텍스트 길이
        """
        self.root = root
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_texts = sorted(glob.glob(os.path.join(root, "*.txt")))

        if phase == 'train':
            self.total_texts = self.total_texts[:int(len(self.total_texts) * 0.7)]
        elif phase == 'val':
            self.total_texts = self.total_texts[int(len(self.total_texts) * 0.7):int(len(self.total_texts) * 0.85)]
        elif phase == 'test':
            self.total_texts = self.total_texts[int(len(self.total_texts) * 0.85):]
        else:
            pass

    def __len__(self):
        return len(self.total_texts)

    def __getitem__(self, idx):
        text_path = self.total_texts[idx]
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # 토크나이저를 사용하여 텍스트를 토큰화
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # squeeze를 통해 배치 차원 제거
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

class MultiModalGlobDataset(Dataset):
    def __init__(self, 
                 text_root: str,
                 image_root: str,
                 phase: str,
                 tokenizer,
                 img_size: int = 128,
                 max_length: int = 512):
        """
        이미지와 텍스트를 함께 처리하는 멀티모달 Dataset 클래스
        
        Args:
            text_root: 텍스트 파일들이 있는 디렉토리 경로
            image_root: 이미지 파일들이 있는 디렉토리 경로
            phase: 'train', 'val', 'test' 중 하나
            tokenizer: 텍스트 토크나이저
            img_size: 이미지 크기
            max_length: 최대 텍스트 길이
        """
        self.text_dataset = TextGlobDataset(text_root, phase, tokenizer, max_length)
        self.image_dataset = GlobDataset(image_root, phase, img_size)
        
        assert len(self.text_dataset) == len(self.image_dataset), \
            "텍스트와 이미지 데이터의 개수가 일치하지 않습니다."

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, idx):
        text_data = self.text_dataset[idx]
        image_data = self.image_dataset[idx]
        
        return {
            'text': text_data,
            'image': image_data
        }

class EmotionTextDataset(Dataset):
    def __init__(self, 
                 csv_path: str,
                 tokenizer,
                 phase: str = 'train',
                 max_length: int = 512,
                 text_column: str = 'SIT',
                 emotion_column: str = 'EMOT',
                 emotion_classes: List[str] = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]):
        """
        감정 분석을 위한 텍스트 데이터셋
        
        Args:
            csv_path: CSV 파일 경로
            tokenizer: 텍스트 토크나이저
            phase: 'train', 'val', 'test' 중 하나
            max_length: 최대 텍스트 길이
            text_column: 텍스트가 있는 컬럼 이름
            emotion_column: 감정 레이블이 있는 컬럼 이름
            emotion_classes: 감정 클래스 리스트
        """
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.emotion_column = emotion_column
        self.emotion_classes = emotion_classes
        
        # 감정 레이블 매핑 (ISEAR 데이터셋 기준)
        self.emotion_mapping = {
            '1': 'joy',
            '2': 'fear',
            '3': 'anger',
            '4': 'sadness',
            '5': 'disgust',
            '6': 'shame',
            '7': 'guilt'
        }
        
        # CSV 파일 로드
        df = pd.read_csv(csv_path, sep='|', encoding='utf-8', on_bad_lines='skip')
        
        # 감정 레이블 변환
        df[emotion_column] = df[emotion_column].astype(str).map(self.emotion_mapping)
        
        # 필요한 컬럼이 있는지 확인
        assert text_column in df.columns, f"{text_column} 컬럼이 없습니다."
        assert emotion_column in df.columns, f"{emotion_column} 컬럼이 없습니다."
        
        # train/val/test 분할
        total_len = len(df)
        if phase == 'train':
            self.df = df[:int(total_len * 0.7)]
        elif phase == 'val':
            self.df = df[int(total_len * 0.7):int(total_len * 0.85)]
        elif phase == 'test':
            self.df = df[int(total_len * 0.85):]
        else:
            self.df = df
            
        # 감정 레이블을 숫자로 변환
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_classes)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_column])
        emotion = str(row[self.emotion_column])
        
        # 텍스트 토큰화
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 감정 레이블을 원-핫 인코딩으로 변환
        emotion_idx = self.emotion_to_idx[emotion]
        emotion_tensor = torch.zeros(len(self.emotion_classes))
        emotion_tensor[emotion_idx] = 1
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'emotion': emotion_tensor,
            'text': text,  # 원본 텍스트도 함께 저장 (시각화 등에 활용)
            'emotion_label': emotion  # 원본 감정 레이블도 함께 저장
        }
    
    def visualize_emotion_distribution(self):
        """감정 분포를 시각화"""
        emotions = self.df[self.emotion_column].value_counts()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=emotions.index, y=emotions.values)
        plt.title('감정 분포')
        plt.xlabel('감정')
        plt.ylabel('빈도')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def get_emotion_statistics(self) -> Dict:
        """감정별 통계 정보 반환"""
        emotion_counts = self.df[self.emotion_column].value_counts()
        total = len(self.df)
        
        stats = {
            'counts': emotion_counts.to_dict(),
            'percentages': (emotion_counts / total * 100).to_dict(),
            'total_samples': total
        }
        return stats
    
    def analyze_text_lengths(self):
        """텍스트 길이 분포 분석"""
        text_lengths = self.df[self.text_column].str.len()
        
        stats = {
            'mean': text_lengths.mean(),
            'median': text_lengths.median(),
            'min': text_lengths.min(),
            'max': text_lengths.max(),
            'std': text_lengths.std()
        }
        
        plt.figure(figsize=(10, 6))
        sns.histplot(text_lengths, bins=50)
        plt.title('텍스트 길이 분포')
        plt.xlabel('텍스트 길이')
        plt.ylabel('빈도')
        plt.tight_layout()
        
        return stats, plt.gcf()
    
    def get_emotion_samples(self, emotion: str, n_samples: int = 5) -> List[str]:
        """특정 감정의 텍스트 샘플 반환"""
        samples = self.df[self.df[self.emotion_column] == emotion][self.text_column].sample(n=min(n_samples, len(self.df)))
        return samples.tolist()
    
    def get_vocabulary_statistics(self, top_n: int = 20):
        """어휘 통계 분석"""
        from collections import Counter
        import re
        
        # 모든 텍스트를 단어로 분리
        words = []
        for text in self.df[self.text_column]:
            words.extend(re.findall(r'\w+', text.lower()))
        
        word_counts = Counter(words)
        
        # 상위 n개 단어와 빈도
        top_words = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        # 감정별 특징적인 단어 분석
        emotion_words = {}
        for emotion in self.emotion_classes:
            emotion_texts = self.df[self.df[self.emotion_column] == emotion][self.text_column]
            emotion_words[emotion] = []
            for text in emotion_texts:
                emotion_words[emotion].extend(re.findall(r'\w+', text.lower()))
            emotion_words[emotion] = Counter(emotion_words[emotion])
        
        return {
            'total_unique_words': len(word_counts),
            'top_words': top_words,
            'emotion_specific_words': emotion_words
        }
