import dearpygui.dearpygui as dpg
from PIL import Image, ImageDraw
from neural_network_torch import NeuralNetwork
from training import load_model
import numpy as np
import cv2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main_net = NeuralNetwork(learning_rate=0.01).to(device=device)

load_model(main_net,"model2.pt")
def open_drawing_window(type,title,size_h_w:tuple = None):
    if size_h_w is None:
        size_h_w = (300, 300)

    points_list = []
    tmp_points_list = []

    drawbox_height, drawbox_width = size_h_w[0], size_h_w[1]

    with dpg.handler_registry(show=True, tag="mouse_handler") as draw_mouse_handler:
        m_wheel = dpg.add_mouse_wheel_handler()
        m_click = dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left)
        m_double_click = dpg.add_mouse_double_click_handler(button=dpg.mvMouseButton_Left)
        m_release = dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left)
        m_drag = dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,threshold=0.0000001)
        m_down = dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Left)
        m_move = dpg.add_mouse_move_handler()

    def _event_handler(sender, data):
        type = dpg.get_item_info(sender)["type"]

        if type == "mvAppItemType::mvMouseReleaseHandler":
            if dpg.is_item_hovered('draw_canvas'):
                points_list.append(tmp_points_list[:])
                if dpg.does_item_exist(item="drawn_lines_layer"):
                    dpg.delete_item(item="drawn_lines_layer")
                if dpg.does_item_exist(item="drawn_lines_layer_tmp"):
                    dpg.delete_item(item="drawn_lines_layer_tmp")
                dpg.add_draw_layer(tag="drawn_lines_layer", parent=canvas)
                for x in points_list:
                    dpg.draw_polyline(points=x,
                                      parent="drawn_lines_layer",
                                      closed=False,
                                      color=(255, 255, 255, 255),
                                      thickness=20)
                tmp_points_list.clear()
        elif type == "mvAppItemType::mvMouseDownHandler" or\
                type == "mvAppItemType::mvMouseDragHandler" or\
                type == "mvAppItemType::mvMouseClickHandler":
            if dpg.is_item_hovered('draw_canvas'):
                cur_mouse_pos = dpg.get_drawing_mouse_pos()
                tmp_points_list.append(tuple(cur_mouse_pos))
                if dpg.does_item_exist(item="drawn_lines_layer_tmp"):
                    dpg.delete_item(item="drawn_lines_layer_tmp")
                if dpg.does_item_exist(item="drawn_lines_layer_tmp"):
                    dpg.delete_item(item="drawn_lines_layer_tmp")
                dpg.add_draw_layer(tag="drawn_lines_layer_tmp", parent=canvas)
                dpg.draw_polyline(points=tmp_points_list,
                                  parent="drawn_lines_layer_tmp",
                                  closed=False,
                                  color=(255, 255, 255, 255),
                                  thickness=20)

    with dpg.window(label="Drawing window", no_close=True, modal=True, tag="draw_window"):
        def erase(sender, data):
            if sender == 'erase_last':
                if points_list:
                    points_list.pop()
                    if dpg.does_item_exist(item="drawn_lines_layer"):
                        dpg.delete_item(item="drawn_lines_layer")

                    dpg.add_draw_layer(tag="drawn_lines_layer", parent=canvas)
                    for x in points_list:
                        dpg.draw_polyline(points=x,
                                          parent="drawn_lines_layer",
                                          closed=False,
                                          color=(255, 255, 255, 255),
                                          thickness=20)
                else:
                    pass

            elif sender == 'erase_all':
                points_list.clear()
                if dpg.does_item_exist(item="drawn_lines_layer"):
                    dpg.delete_item(item="drawn_lines_layer")


        def detect(sender, data):
            if sender == 'detect_tag':
                output_img = Image.new(mode="RGB", size=(drawbox_width, drawbox_height))
                draw = ImageDraw.Draw(output_img)
                for y in points_list:
                    draw.line(y, None, 40, None)
                dpg.delete_item('class_text')
                dpg.add_text(default_value="Detected number:"+str(np.argmax(a=main_net(torch.tensor(cv2.resize(np.array(output_img.convert(mode="L")),(28,28)).flatten()/255).float().to(device=device)).cpu().detach().numpy())), parent='draw_window',tag='class_text')

        for handler in dpg.get_item_children("mouse_handler", 1):
            dpg.set_item_callback(handler, _event_handler)

        with dpg.group(tag='cnt_btns', horizontal=True, parent="draw_window") as buttons:
            dpg.add_button(label='Erase last', callback=erase, tag='erase_last')
            dpg.add_spacer(width=30)
            dpg.add_button(label='Erase all', callback=erase, tag='erase_all')
            dpg.add_spacer(width=30)
            dpg.add_button(label='Detect', callback=detect, tag='detect_tag')

        with dpg.child_window(label="canvas_border", tag='canvas_border', width=drawbox_width+10,
                              height=drawbox_height+10, border=True, no_scrollbar=True, parent='draw_window'):
            with dpg.drawlist(width=drawbox_width, height=drawbox_height,
                              tag="draw_canvas", parent="canvas_border") as canvas:
                pass

if __name__ == '__main__':
    dpg.create_context()

    open_drawing_window(type='drawing',
                        title='technician',
                        size_h_w=None
                        )

    dpg.create_viewport(title='Classification', width=400, height=500)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()