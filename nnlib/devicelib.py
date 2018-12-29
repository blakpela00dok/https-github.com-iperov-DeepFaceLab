from .pynvml import *

class devicelib:
    class Config():
        force_best_gpu_idx = -1
        multi_gpu = False
        force_gpu_idxs = None
        choose_worst_gpu = False
        gpu_idxs = []
        gpu_total_vram_gb = 0
        allow_growth = True
        float16 = False
        cpu_only = False

        def __init__ (self, force_best_gpu_idx = -1,
                            multi_gpu = False,
                            force_gpu_idxs = None,
                            choose_worst_gpu = False,
                            allow_growth = True,
                            float16 = False,
                            cpu_only = False,
                            **in_options):

            self.float16 = float16
            if cpu_only or not devicelib.hasNVML():
                self.cpu_only = True
            else:
                self.force_best_gpu_idx = force_best_gpu_idx
                self.multi_gpu = multi_gpu
                self.force_gpu_idxs = force_gpu_idxs
                self.choose_worst_gpu = choose_worst_gpu
                self.allow_growth = allow_growth

                self.gpu_idxs = []
                if force_gpu_idxs is not None:
                    for idx in force_gpu_idxs.split(','):
                        idx = int(idx)
                        if devicelib.isValidDeviceIdx(idx):
                            self.gpu_idxs.append(idx)
                else:
                    gpu_idx = force_best_gpu_idx if (force_best_gpu_idx >= 0 and devicelib.isValidDeviceIdx(force_best_gpu_idx)) else devicelib.getBestDeviceIdx() if not choose_worst_gpu else devicelib.getWorstDeviceIdx()
                    if gpu_idx != -1:
                        if self.multi_gpu:
                            self.gpu_idxs = devicelib.getDeviceIdxsEqualModel( gpu_idx )
                            if len(self.gpu_idxs) <= 1:
                                self.multi_gpu = False
                        else:
                            self.gpu_idxs = [gpu_idx]

                if len(self.gpu_idxs) == 0:
                    self.cpu_only = True
                else:
                    self.cpu_only = False
                    self.gpu_total_vram_gb = devicelib.getDeviceVRAMTotalGb ( self.gpu_idxs[0] )

    @staticmethod
    def hasNVML():
        try:
            nvmlInit()
            nvmlShutdown()
        except:
            return False
        return True

    @staticmethod
    def getDevicesWithAtLeastFreeMemory(freememsize):
        result = []
        try:
            nvmlInit()
            for i in range(0, nvmlDeviceGetCount() ):
                handle = nvmlDeviceGetHandleByIndex(i)
                memInfo = nvmlDeviceGetMemoryInfo( handle )
                if (memInfo.total - memInfo.used) >= freememsize:
                    result.append (i)
            nvmlShutdown()
        except:
            pass
        return result

    @staticmethod
    def getDevicesWithAtLeastTotalMemoryGB(totalmemsize_gb):
        result = []
        try:
            nvmlInit()
            for i in range(0, nvmlDeviceGetCount() ):
                handle = nvmlDeviceGetHandleByIndex(i)
                memInfo = nvmlDeviceGetMemoryInfo( handle )
                if (memInfo.total) >= totalmemsize_gb*1024*1024*1024:
                    result.append (i)
            nvmlShutdown()
        except:
            pass
        return result

    @staticmethod
    def getAllDevicesIdxsList ():
        result = []
        try:
            nvmlInit()
            result = [ i for i in range(0, nvmlDeviceGetCount() ) ]
            nvmlShutdown()
        except:
            pass
        return result

    @staticmethod
    def getDeviceVRAMFree (idx):
        result = 0
        try:
            nvmlInit()
            if idx < nvmlDeviceGetCount():
                handle = nvmlDeviceGetHandleByIndex(idx)
                memInfo = nvmlDeviceGetMemoryInfo( handle )
                result = (memInfo.total - memInfo.used)
            nvmlShutdown()
        except:
            pass
        return result

    @staticmethod
    def getDeviceVRAMTotalGb (idx):
        result = 0
        try:
            nvmlInit()
            if idx < nvmlDeviceGetCount():
                handle = nvmlDeviceGetHandleByIndex(idx)
                memInfo = nvmlDeviceGetMemoryInfo( handle )
                result = memInfo.total / (1024*1024*1024)
            nvmlShutdown()
            result = round(result)
        except:
            pass
        return result

    @staticmethod
    def getBestDeviceIdx():
        idx = -1
        try:
            nvmlInit()
            idx_mem = 0
            for i in range(0, nvmlDeviceGetCount() ):
                handle = nvmlDeviceGetHandleByIndex(i)
                memInfo = nvmlDeviceGetMemoryInfo( handle )
                if memInfo.total > idx_mem:
                    idx = i
                    idx_mem = memInfo.total

            nvmlShutdown()
        except:
            pass
        return idx

    @staticmethod
    def getWorstDeviceIdx():
        idx = -1
        try:
            nvmlInit()

            idx_mem = sys.maxsize
            for i in range(0, nvmlDeviceGetCount() ):
                handle = nvmlDeviceGetHandleByIndex(i)
                memInfo = nvmlDeviceGetMemoryInfo( handle )
                if memInfo.total < idx_mem:
                    idx = i
                    idx_mem = memInfo.total

            nvmlShutdown()
        except:
            pass
        return idx

    @staticmethod
    def isValidDeviceIdx(idx):
        result = False
        try:
            nvmlInit()
            result = (idx < nvmlDeviceGetCount())
            nvmlShutdown()
        except:
            pass
        return result

    @staticmethod
    def getDeviceIdxsEqualModel(idx):
        result = []
        try:
            nvmlInit()
            idx_name = nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(idx)).decode()

            for i in range(0, nvmlDeviceGetCount() ):
                if nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)).decode() == idx_name:
                    result.append (i)

            nvmlShutdown()
        except:
            pass
        return result

    @staticmethod
    def getDeviceName (idx):
        result = ''
        try:
            nvmlInit()
            if idx < nvmlDeviceGetCount():
                result = nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(idx)).decode()
            nvmlShutdown()
        except:
            pass
        return result
