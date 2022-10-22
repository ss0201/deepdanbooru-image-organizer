import hashlib
import json
from typing import Union, cast

import sqlalchemy
from sqlalchemy import Column, String, Text, insert, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Image(Base):
    __tablename__ = "images"
    md5 = cast(str, Column(String(32), primary_key=True))
    tag_evaluation = cast(str, Column(Text))


class DDCache:
    engine: sqlalchemy.engine.Engine

    def __init__(self, db_path: str):
        self.engine = sqlalchemy.create_engine(f"sqlite:///{db_path}", future=True)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()

    def get_cached_evaluations(self, image_path: str) -> Union[list[float], None]:
        md5 = self.__get_md5(image_path)
        image: Union[Image, None] = self.session.execute(
            select(Image).where(Image.md5 == md5)
        ).scalars().first()
        if image:
            return json.loads(image.tag_evaluation)
        else:
            return None

    def cache_evaluations(self, image_path: str, evaluations: list[float]) -> None:
        md5 = self.__get_md5(image_path)
        self.session.execute(
            insert(Image).values(md5=md5, tag_evaluation=json.dumps(evaluations))
        )
        self.session.commit()

    def __get_md5(self, image_path: str) -> str:
        md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()
