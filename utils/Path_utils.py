from operator import attrgetter
from pathlib import Path
from os import scandir
from typing import List, Optional, Callable

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def get_image_paths(dir_path: str, image_extensions: List[str] = IMAGE_EXTENSIONS) -> List[str]:
    dir_path = Path (dir_path)

    result = []
    if dir_path.exists():
        for x in scandir(str(dir_path)):
            if any([x.name.lower().endswith(ext) for ext in image_extensions]):
                result.append(x.path)
    return sorted(result)


def get_image_unique_filestem_paths(dir_path: str, verbose_print_func: Optional[Callable[[str], None]] = None) -> List[str]:
    result = get_image_paths(dir_path)
    result_dup = set()

    for f in result[:]:
        f_stem = Path(f).stem
        if f_stem in result_dup:
            result.remove(f)
            if verbose_print_func is not None:
                verbose_print_func ("Duplicate filenames are not allowed, skipping: %s" % Path(f).name )
            continue
        result_dup.add(f_stem)

    return sorted(result)
    

def get_file_paths(dir_path: str) -> List[str]:
    dir_path = Path(dir_path)
    if dir_path.exists():
        return sorted([x.path for x in scandir(str(dir_path)) if x.is_file()])
    return []


def get_all_dir_names(dir_path: str) -> List[str]:
    dir_path = Path(dir_path)
    if dir_path.exists():
        return sorted([x.name for x in scandir(str(dir_path)) if x.is_dir()])
    return []


def get_all_dir_names_startswith(dir_path: str, startswith: str) -> List[str]:
    dir_path = Path (dir_path)
    startswith = startswith.lower()

    result = []
    if dir_path.exists():
        for x in scandir(str(dir_path)):
            if x.name.lower().startswith(startswith):
                result.append ( x.name[len(startswith):] )
    return sorted(result)


def get_first_file_by_stem(dir_path: str, stem: str, exts: List[str] = None) -> Optional[Path]:
    dir_path = Path (dir_path)
    stem = stem.lower()

    if dir_path.exists():
        for x in sorted(scandir(str(dir_path)), key=attrgetter('name')):
            if not x.is_file():
                continue
            xp = Path(x.path)
            if xp.stem.lower() == stem and (exts is None or xp.suffix.lower() in exts):
                return xp

    return None


def move_all_files(src_dir_path: str, dst_dir_path: str) -> None:
    paths = get_file_paths(src_dir_path)
    for p in paths:
        p = Path(p)
        p.rename ( Path(dst_dir_path) / p.name )
        

def delete_all_files(dir_path: str) -> None:
    paths = get_file_paths(dir_path)
    for p in paths:
        p = Path(p)
        p.unlink()
