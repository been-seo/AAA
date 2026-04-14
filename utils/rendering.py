"""Pygame 텍스트 렌더링 유틸리티"""
import pygame


def render_text_with_simple_outline(font, text, text_color, outline_color):
    """두꺼운 외곽선이 있는 텍스트 Surface 생성 (8방향)"""
    try:
        base = font.render(text, True, text_color)
        outline = font.render(text, True, outline_color)
    except pygame.error:
        return pygame.Surface((10, 10), pygame.SRCALPHA)
    w, h = base.get_size()
    result = pygame.Surface((w + 4, h + 4), pygame.SRCALPHA)
    # 8방향 + 거리2 외곽선
    for ox, oy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1),
                   (-2,0),(2,0),(0,-2),(0,2)]:
        result.blit(outline, (ox + 2, oy + 2))
    result.blit(base, (2, 2))
    return result
