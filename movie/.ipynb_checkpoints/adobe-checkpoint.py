import asyncio
from pathlib import Path
from aerender import AERenderWrapper

if __name__ == '__main__':
    AERENDER_FULLPATH = Path('C:/Program Files/Adobe/Adobe After Effects CC 2019/Support Files/aerender')
    aerender = AERenderWrapper(exe_path=AERENDER_FULLPATH)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(aerender.run(
        project_path=Path(r"E:\graphics\gg\Graphic-Pack\Montage Graphic Pack\Montage Graphic Pack CC2019 OR UP\presets\AE\Montage Graphic Pack\Animated Shapes\Flash FX\07.aep"),
        comp_name='Comp 1',
        output_path=Path('proj1.avi'),
    ))
    loop.close()



