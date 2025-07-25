"""
Responsive Image Renderer - Click-to-Enlarge with Raw PNG Download

Image rendering with:
- Responsive sizing
- Click to enlarge functionality
- Image optimization
- Raw PNG download always available
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional, Union
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)


class ResponsiveImageRenderer:
    """Responsive image renderer with advanced features"""
    
    def __init__(self):
        """Initialize image renderer"""
        self.max_display_width = 800
        self.thumbnail_size = (300, 300)
        
    def render_image(self,
                    image_data: Union[Image.Image, bytes, str],
                    title: Optional[str] = None,
                    caption: Optional[str] = None,
                    enable_enlarge: bool = True,
                    enable_download: bool = True) -> Dict[str, Any]:
        """
        Render responsive image with features
        
        Returns:
            Dict with 'raw_png' for download
        """
        try:
            # Convert to PIL Image
            image = self._convert_to_pil(image_data)
            
            if title:
                st.markdown(f"### 🖼️ {title}")
            
            # Image info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("크기", f"{image.width} × {image.height}")
            with col2:
                st.metric("형식", image.format or "Unknown")
            with col3:
                file_size = len(self._image_to_bytes(image)) / 1024
                st.metric("파일 크기", f"{file_size:.1f} KB")
            
            # Responsive display
            display_image = self._optimize_for_display(image)
            
            # Main image display
            if enable_enlarge:
                # Create columns for centering
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    # Clickable image with modal
                    clicked = st.image(
                        display_image,
                        caption=caption,
                        use_column_width=True
                    )
                    
                    if st.button("🔍 이미지 확대", use_container_width=True):
                        self._show_enlarged_image(image, title)
            else:
                st.image(display_image, caption=caption, use_column_width=True)
            
            # Download button
            if enable_download:
                png_data = self._image_to_bytes(image, format='PNG')
                
                st.download_button(
                    label="⬇️ PNG 다운로드",
                    data=png_data,
                    file_name=f"{title or 'image'}.png",
                    mime="image/png",
                    key=f"download_img_{hash(str(image_data)[:50])}"
                )
            
            # Image analysis
            with st.expander("📊 이미지 분석", expanded=False):
                self._analyze_image(image)
            
            return {
                'raw_png': png_data if enable_download else None,
                'width': image.width,
                'height': image.height,
                'format': image.format
            }
            
        except Exception as e:
            logger.error(f"Error rendering image: {str(e)}")
            st.error(f"이미지 렌더링 오류: {str(e)}")
            return {'raw_png': None}
    
    def _convert_to_pil(self, image_data: Union[Image.Image, bytes, str]) -> Image.Image:
        """Convert various image formats to PIL Image"""
        if isinstance(image_data, Image.Image):
            return image_data
        
        elif isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data))
        
        elif isinstance(image_data, str):
            # Base64 encoded string
            if image_data.startswith('data:image'):
                # Extract base64 part
                base64_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(image_bytes))
            else:
                # Try as file path
                return Image.open(image_data)
        
        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}")
    
    def _optimize_for_display(self, image: Image.Image) -> Image.Image:
        """Optimize image for web display"""
        # Convert RGBA to RGB if needed (for JPEG compatibility)
        if image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        
        # Resize if too large
        if image.width > self.max_display_width:
            ratio = self.max_display_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((self.max_display_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def _image_to_bytes(self, image: Image.Image, format: str = 'PNG') -> bytes:
        """Convert PIL Image to bytes"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    def _show_enlarged_image(self, image: Image.Image, title: Optional[str] = None) -> None:
        """Show enlarged image in modal-like display"""
        with st.container():
            st.markdown("---")
            st.markdown(f"#### 🔍 확대 보기: {title or '이미지'}")
            
            # Full size image
            st.image(image, use_column_width=True)
            
            # Close button
            if st.button("✖️ 닫기"):
                st.experimental_rerun()
            
            st.markdown("---")
    
    def _analyze_image(self, image: Image.Image) -> None:
        """Analyze and display image properties"""
        # Basic properties
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**기본 정보**")
            st.write(f"- 모드: {image.mode}")
            st.write(f"- 형식: {image.format or 'Unknown'}")
            st.write(f"- 가로세로 비율: {image.width/image.height:.2f}")
        
        with col2:
            st.write("**색상 정보**")
            if image.mode in ['RGB', 'RGBA']:
                # Get color statistics
                pixels = list(image.getdata())
                if len(pixels) > 0:
                    # Sample pixels for performance
                    sample_size = min(1000, len(pixels))
                    import random
                    sample_pixels = random.sample(pixels, sample_size)
                    
                    # Calculate average color
                    avg_r = sum(p[0] for p in sample_pixels) / sample_size
                    avg_g = sum(p[1] for p in sample_pixels) / sample_size
                    avg_b = sum(p[2] for p in sample_pixels) / sample_size
                    
                    st.write(f"- 평균 색상: RGB({avg_r:.0f}, {avg_g:.0f}, {avg_b:.0f})")
                    
                    # Show color swatch
                    color_swatch = Image.new('RGB', (50, 50), 
                                           (int(avg_r), int(avg_g), int(avg_b)))
                    st.image(color_swatch, width=50)
    
    def render_image_gallery(self,
                           images: list,
                           titles: Optional[list] = None,
                           columns: int = 3) -> None:
        """Render multiple images in a gallery layout"""
        try:
            st.markdown("### 🖼️ 이미지 갤러리")
            
            # Create grid
            cols = st.columns(columns)
            
            for idx, image_data in enumerate(images):
                col_idx = idx % columns
                
                with cols[col_idx]:
                    # Convert to PIL
                    image = self._convert_to_pil(image_data)
                    
                    # Create thumbnail
                    thumbnail = image.copy()
                    thumbnail.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                    
                    # Display
                    title = titles[idx] if titles and idx < len(titles) else f"이미지 {idx + 1}"
                    st.image(thumbnail, caption=title, use_column_width=True)
                    
                    # Enlarge button
                    if st.button(f"🔍 확대", key=f"enlarge_{idx}"):
                        self._show_enlarged_image(image, title)
                        
        except Exception as e:
            logger.error(f"Error rendering gallery: {str(e)}")
            st.error(f"갤러리 렌더링 오류: {str(e)}")
    
    def render_image_comparison(self,
                              image1: Union[Image.Image, bytes, str],
                              image2: Union[Image.Image, bytes, str],
                              label1: str = "이전",
                              label2: str = "이후") -> None:
        """Render side-by-side image comparison"""
        try:
            st.markdown("### 🔄 이미지 비교")
            
            # Convert to PIL
            img1 = self._convert_to_pil(image1)
            img2 = self._convert_to_pil(image2)
            
            # Optimize for display
            img1_display = self._optimize_for_display(img1)
            img2_display = self._optimize_for_display(img2)
            
            # Side by side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{label1}**")
                st.image(img1_display, use_column_width=True)
                st.caption(f"{img1.width} × {img1.height}")
            
            with col2:
                st.markdown(f"**{label2}**")
                st.image(img2_display, use_column_width=True)
                st.caption(f"{img2.width} × {img2.height}")
            
            # Difference analysis
            if img1.size == img2.size:
                with st.expander("📊 차이 분석", expanded=False):
                    self._analyze_image_difference(img1, img2)
                    
        except Exception as e:
            logger.error(f"Error comparing images: {str(e)}")
            st.error(f"이미지 비교 오류: {str(e)}")
    
    def _analyze_image_difference(self, img1: Image.Image, img2: Image.Image) -> None:
        """Analyze difference between two images"""
        try:
            from PIL import ImageChops
            
            # Calculate difference
            diff = ImageChops.difference(img1.convert('RGB'), img2.convert('RGB'))
            
            # Get bounding box of changes
            bbox = diff.getbbox()
            
            if bbox:
                st.write(f"**변경된 영역:** {bbox}")
                
                # Highlight differences
                diff_highlight = Image.new('RGB', img1.size, (255, 255, 255))
                diff_highlight.paste(diff, bbox)
                
                st.image(diff_highlight, caption="차이점 하이라이트", use_column_width=True)
            else:
                st.info("두 이미지가 동일합니다.")
                
        except Exception as e:
            logger.error(f"Error analyzing difference: {str(e)}")