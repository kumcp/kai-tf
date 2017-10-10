from readerwriter_detection_base import KReaderWriterBase
import csv
import os

class KReaderWriter(KReaderWriterBase):
    def get_bounding_boxes(self,image_name):
        bb_path = os.path.join(self.path,"boxes",image_name+".csv")
        
        with open(bb_path,'rb') as file:
            reader = csv.reader(file,delimiter=",")
            bbxmin = []
            bbymin = []
            bbxmax = []
            bbymax = []
            label = []
            hidden = []
            for line in reader:
                bbxmin.append(int(line[0]))
                bbymin.append(int(line[1]))
                bbxmax.append(int(line[2]))
                bbymax.append(int(line[3]))
                label.append(int(line[4]))
                item_in_row = []
                for hidden_item in range(5,len(line)):
                    item_in_row.append(line[hidden_item])
                if len(item_in_row) > 0:
                    hidden.append(item_in_row)
                

        result = {
            "bounding_box_xmin": bbxmin,
            "bounding_box_ymin": bbymin,
            "bounding_box_xmax": bbxmax,
            "bounding_box_ymax": bbymax,
            "label_int": label,
        }

        if len(hidden) > 0:
            result["hidden"] = hidden

        return result