
omnigen2_1=$(ls /mnt/localssd/spot_edit/spotframe_real/omnigen2_edited_videos/ | wc -l)
bagel_1=$(ls /mnt/localssd/spot_edit/spotframe_real/bagel_edited_videos/ | wc -l)
uno_1=$(ls /mnt/localssd/spot_edit/spotframe_real/uno_edited_videos/ | wc -l)
omnigen_1=$(ls /mnt/localssd/spot_edit/spotframe_real/omnigen_edited_videos/ | wc -l)
emu2_1=$(ls /mnt/localssd/spot_edit/spotframe_real/emu2_edited_videos/ | wc -l)

sleep 0

omnigen2_2=$(ls /mnt/localssd/spot_edit/spotframe_real/omnigen2_edited_videos/ | wc -l)
bagel_2=$(ls /mnt/localssd/spot_edit/spotframe_real/bagel_edited_videos/ | wc -l)
uno_2=$(ls /mnt/localssd/spot_edit/spotframe_real/uno_edited_videos/ | wc -l)
omnigen_2=$(ls /mnt/localssd/spot_edit/spotframe_real/omnigen_edited_videos/ | wc -l)
emu2_2=$(ls /mnt/localssd/spot_edit/spotframe_real/emu2_edited_videos/ | wc -l)



echo omnigen2 $omnigen2_1 $omnigen2_2

echo bagel_edited $bagel_1 $bagel_2

echo uno_edited $uno_1 $uno_2

echo omnigen $omnigen_1 $omnigen_2

echo emu2 $emu2_1 $emu2_2
