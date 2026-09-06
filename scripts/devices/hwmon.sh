#!/system/bin/sh
# hwmon.sh — echantillon temps reel CPU/GPU/NPU/RAM/batt/thermique (une passe)
# Sortie : lignes cle=valeur, parsees par l'app. Boucle a 1Hz cote app.
#
# REECRIT (2026-09-06) suite a la demande explicite "on ne voit pas tous les
# coeurs, les % et temperature de tous" — version precedente ne donnait que
# 3 coeurs fixes (cpu0/4/8) et 4 zones thermiques nommees en dur. Cette
# version boucle sur TOUS les coeurs presents et TOUTES les zones
# thermiques du device, generique (marche sur n'importe quel Snapdragon,
# pas seulement SM8850).
p=/sys/devices/system/cpu

# Detecte dynamiquement les coeurs presents (cpu0, cpu1, ... cpuN)
cores=""
for d in $p/cpu[0-9]*; do
  [ -d "$d" ] || continue
  n=$(basename "$d" | sed 's/cpu//')
  cores="$cores $n"
done

# CPU % GLOBAL + PAR COEUR via /proc/stat (2 lectures espacees de 0.5s,
# une ligne "cpuN ..." par coeur en plus de la ligne agregee "cpu ...")
s1=$(cat /proc/stat)
sleep 0.5
s2=$(cat /proc/stat)

calc_pct() {
  # $1 = ligne stat t1, $2 = ligne stat t2 (deja "cpu... u n s i ...")
  # BUG REEL trouve et corrige : `set --` a la premiere ligne ECRASE $1/$2
  # (les arguments de la fonction elle-meme), donc la deuxieme utilisation
  # de $1 plus bas ne pointait plus sur la ligne t1 d'origine mais sur un
  # token de la ligne t2 deja "set --"-ee -> arithmetique avec des champs
  # vides -> erreur shell "++ requires lvalue" et pct toujours vide.
  # Fix : copier les deux arguments dans des variables AVANT tout `set --`.
  line1="$1"
  line2="$2"
  set -- $line2
  tot2=$(( $2+$3+$4+$5+$6+$7+$8+$9+${10} ))
  idle2=$(( $5+$6 ))
  set -- $line1
  tot1=$(( $2+$3+$4+$5+$6+$7+$8+$9+${10} ))
  idle1=$(( $5+$6 ))
  dt=$((tot2-tot1)); di=$((idle2-idle1))
  if [ "$dt" -gt 0 ]; then echo $(( (100*(dt-di))/dt )); else echo 0; fi
}

l1=$(echo "$s1" | grep '^cpu ')
l2=$(echo "$s2" | grep '^cpu ')
echo "cpu_pct=$(calc_pct "$l1" "$l2")"

for n in $cores; do
  l1n=$(echo "$s1" | grep "^cpu$n ")
  l2n=$(echo "$s2" | grep "^cpu$n ")
  if [ -n "$l1n" ] && [ -n "$l2n" ]; then
    echo "cpu${n}_pct=$(calc_pct "$l1n" "$l2n")"
  fi
  echo "cpu${n}_freq=$(cat $p/cpu$n/cpufreq/scaling_cur_freq 2>/dev/null)"
done

# GPU : clk + busy%
gclk=$(cat /sys/class/kgsl/kgsl-3d0/gpuclk 2>/dev/null)
gbusy=$(cat /sys/class/kgsl/kgsl-3d0/gpubusy 2>/dev/null)
echo "gpu_freq=$gclk"
set -- $gbusy
if [ -n "$2" ] && [ "$2" -gt 0 ]; then gpct=$(( $1*100/$2 )); else gpct=0; fi
echo "gpu_pct=$gpct"

# Thermique : TOUTES les zones presentes (pas une liste fixe) — nom de zone
# nettoye (slash/tiret -> underscore) comme cle, temperature en degres C.
for z in /sys/class/thermal/thermal_zone*; do
  [ -d "$z" ] || continue
  t=$(cat "$z/type" 2>/dev/null)
  v=$(cat "$z/temp" 2>/dev/null)
  [ -n "$t" ] && [ -n "$v" ] || continue
  c=$((v/1000))
  # Filtre les capteurs sentinelles/absents (ex -273°C = zero absolu,
  # certains modems/radios non actifs le rapportent litteralement)
  [ "$c" -le -250 ] && continue
  key=$(echo "$t" | tr '/-' '__')
  echo "temp_${key}=$c"
done

# RAM
mav=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
mtot=$(grep MemTotal /proc/meminfo | awk '{print $2}')
echo "ram_pct=$(( (mtot-mav)*100/mtot ))"
echo "ram_avail_mb=$((mav/1024))"

# batterie
echo "batt_pct=$(cat /sys/class/power_supply/battery/capacity 2>/dev/null)"
cur=$(cat /sys/class/power_supply/battery/current_now 2>/dev/null)
vol=$(cat /sys/class/power_supply/battery/voltage_now 2>/dev/null)
if [ -n "$cur" ] && [ -n "$vol" ]; then
  w=$(( cur*vol/1000000000000 )); [ "$w" -lt 0 ] && w=$((-w))
  echo "batt_w=$w"
fi
