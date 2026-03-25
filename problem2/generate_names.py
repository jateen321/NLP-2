"""
generate_names.py - Generate 1000 Indian Names
================================================
Generates a diverse dataset of 1000 Indian names from common
first name components across different regions and genders.

Usage:
    python problem2/generate_names.py

Output:
    problem2/TrainingNames.txt  — one name per line
"""

import random
import os

random.seed(42)

# ──────────────────────────────────────────────
# Diverse Indian Name Lists (by region/gender)
# ──────────────────────────────────────────────

# North Indian Male Names
NORTH_MALE = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Sai",
    "Arnav", "Dhruv", "Kabir", "Ritvik", "Rudra", "Shaurya", "Ayaan",
    "Atharv", "Darsh", "Harsh", "Ishaan", "Krishna", "Lakshay",
    "Manav", "Nakul", "Om", "Pranav", "Raghav", "Sahil", "Tanish",
    "Utkarsh", "Varun", "Yash", "Abhinav", "Bharat", "Chirag",
    "Deepak", "Gaurav", "Hemant", "Jatin", "Karan", "Lalit", "Mohit",
    "Nikhil", "Pankaj", "Ravi", "Sunil", "Tarun", "Vikas", "Ankit",
    "Rohit", "Ashish", "Prashant", "Rahul", "Sanjay", "Vijay",
    "Manoj", "Rajesh", "Suresh", "Ramesh", "Dinesh", "Naresh",
    "Rakesh", "Mukesh", "Hitesh", "Ritesh", "Nimesh", "Paresh",
    "Alpesh", "Jayesh", "Nilesh", "Umesh", "Yogesh", "Ganesh",
    "Sudhir", "Satish", "Girish", "Harish", "Manish", "Rajat",
    "Sumit", "Amit", "Namit", "Vinit", "Puneet", "Pradeep",
    "Sandeep", "Kuldeep", "Pardeep", "Randeep", "Jaideep", "Navdeep",
    "Gurdeep", "Amritpal", "Harpal", "Jaspal", "Rajpal", "Kartar",
    "Balraj", "Indrajit", "Amarjit", "Surjit", "Daljit", "Manjit",
    "Ranjit", "Jagjit", "Paramjit", "Gurpreet", "Harpreet", "Manpreet",
    "Navpreet", "Jaspreet", "Simranjeet", "Sukhwinder", "Harvinder",
    "Devender", "Narinder", "Joginder", "Mahender", "Rajinder",
    "Surinder", "Virendra", "Dhirendra", "Gyanendra", "Nagendra",
    "Ravindra", "Davinder", "Bhupinder", "Lakhwinder", "Gurdas",
    "Angad", "Arjan", "Avtar", "Baldev",
]

# North Indian Female Names
NORTH_FEMALE = [
    "Aanya", "Diya", "Myra", "Anaya", "Aadhya", "Pihu", "Pari",
    "Anika", "Kavya", "Sara", "Navya", "Ira", "Ahana", "Kiara",
    "Mishka", "Saanvi", "Ishani", "Kyra", "Shanaya", "Trisha",
    "Riya", "Priya", "Neha", "Pooja", "Nisha", "Swati", "Pallavi",
    "Preeti", "Anjali", "Deepika", "Shruti", "Anushka", "Divya",
    "Komal", "Sonia", "Shweta", "Rashmi", "Meenakshi", "Sunita",
    "Anita", "Reena", "Seema", "Shashi", "Rekha", "Suman",
    "Kiran", "Pushpa", "Savita", "Usha", "Kamla", "Sarla",
    "Nirmal", "Kusum", "Manju", "Indu", "Madhu", "Pratibha",
    "Vandana", "Archana", "Rachna", "Sapna", "Sudha", "Meena",
    "Lata", "Asha", "Ritu", "Geeta", "Sita", "Radha",
    "Aishwarya", "Bhavna", "Chhaya", "Darshana", "Ekta", "Garima",
    "Himani", "Jyoti", "Kalpana", "Leela", "Mamta", "Nandini",
    "Padma", "Ranjana", "Shobha", "Tara", "Uma", "Veena",
    "Yamini", "Zara", "Aditi", "Bhagya", "Chandra", "Damini",
    "Gauri", "Hema", "Indira", "Janaki", "Kamini", "Lalita",
    "Mohini", "Nirmala", "Parvati", "Rajani", "Shakti", "Tulsi",
    "Urmila", "Vasanti", "Harpreet", "Gurpreet", "Jaspreet",
    "Simran", "Navneet", "Amanpreet", "Sukhpreet", "Manmeet",
]

# South Indian Male Names
SOUTH_MALE = [
    "Arun", "Bala", "Chandran", "Dinesh", "Ezhil", "Ganesh",
    "Hari", "Karthik", "Mohan", "Naren", "Prabhu", "Raghu",
    "Senthil", "Thiru", "Velu", "Anand", "Bharath", "Chetan",
    "Deepan", "Gokul", "Harish", "Jagadeesh", "Karthikeyan",
    "Manikandan", "Naveen", "Pravin", "Rajkumar", "Saravanan",
    "Thirumalai", "Vijayakumar", "Ashok", "Balaji", "Dhanush",
    "Gautam", "Srinivas", "Surya", "Venkatesh", "Prasad",
    "Ramakrishnan", "Subramanian", "Sundaram", "Raghunathan",
    "Narayanan", "Krishnamurthy", "Venkataraman", "Padmanabhan",
    "Ranganathan", "Shanmugam", "Palaniswamy", "Thyagarajan",
    "Gopalakrishnan", "Lakshminarayanan", "Seshadri", "Vasudevan",
    "Mahadevan", "Sivaraman", "Chandrasekar", "Muthukrishnan",
    "Selvaraj", "Arumugam", "Murugan", "Kumaran",
    "Aravind", "Lokesh", "Nithish", "Sathish", "Vignesh",
    "Ramesh", "Suresh", "Mahesh", "Kamalesh", "Yogesh",
]

# South Indian Female Names
SOUTH_FEMALE = [
    "Ammu", "Bhavani", "Chitra", "Devika", "Geetha", "Hamsini",
    "Janani", "Keerthi", "Lakshmi", "Meera", "Nithya", "Padma",
    "Preethi", "Revathi", "Saranya", "Thanga", "Vasuki", "Yamuna",
    "Anitha", "Brindha", "Dharani", "Gomathi", "Hemalatha",
    "Jayanthi", "Kanchana", "Malathi", "Nandhini", "Pavithra",
    "Rani", "Sangeetha", "Tamilselvi", "Varalakshmi", "Aishwarya",
    "Deepa", "Kavitha", "Lalitha", "Mala", "Radha",
    "Shanthi", "Uma", "Vijaya", "Kamala", "Saraswathi",
    "Parvathi", "Rukmani", "Meenakshi", "Gomathy", "Soundarya",
    "Suganya", "Thenmozhi", "Iswarya", "Ponni", "Vani",
    "Kala", "Mathi", "Selvi", "Rani", "Devi",
]

# Bengali Names
BENGALI_MALE = [
    "Abhijit", "Arijit", "Arnab", "Ayan", "Debashish", "Dipankar",
    "Indranil", "Joy", "Kaushik", "Mainak", "Partha", "Prosenjit",
    "Rajdeep", "Sabyasachi", "Sourav", "Subhajit", "Suman", "Tanmoy",
    "Tapas", "Uttam", "Bikram", "Chiranjib", "Gautam", "Arpan",
    "Shubham", "Debanjan", "Anirban", "Saptarshi", "Rupam", "Niloy",
    "Subrata", "Arup", "Bhaskar", "Pradip", "Sudip",
    "Ashim", "Bibhas", "Chanchal", "Dilip", "Gopal",
]

BENGALI_FEMALE = [
    "Aditi", "Arpita", "Chandrima", "Deboleena", "Gargi", "Ishita",
    "Jayeeta", "Kasturi", "Moumita", "Nandita", "Paromita", "Rituparna",
    "Sayantani", "Sreyashi", "Trisha", "Madhurima", "Laboni",
    "Rimjhim", "Antara", "Bonani", "Sohini", "Tanushree", "Ankita",
    "Payel", "Swastika", "Oindrila", "Poulami", "Brishti", "Sharmila",
    "Pallabi", "Raima", "Priyanka", "Jhilik", "Keya",
]

# Gujarati Names
GUJARATI_MALE = [
    "Aakash", "Brijesh", "Chetan", "Darshan", "Falgun", "Gaurang",
    "Hardik", "Jignesh", "Ketan", "Manan", "Niral", "Piyush",
    "Rushabh", "Sagar", "Tejas", "Utsav", "Vatsal", "Yagnesh",
    "Parth", "Dhaval", "Hiren", "Krunal", "Mitesh", "Nishant",
    "Bhavin", "Chintan", "Dharmesh", "Jigar", "Keyur", "Meet",
    "Ronak", "Shubh", "Kartik", "Kunal", "Mihir",
]

GUJARATI_FEMALE = [
    "Bhumi", "Dhara", "Falguni", "Hetal", "Jagruti", "Kruti",
    "Meghna", "Nidhi", "Payal", "Riddhi", "Siddhi", "Toral",
    "Vaishali", "Zankhana", "Ami", "Bina", "Chandni", "Dipti",
    "Heena", "Isha", "Khushbu", "Mansi", "Nehal", "Prachi",
    "Ruchi", "Shilpa", "Smriti", "Twinkle", "Vrunda", "Drashti",
]

# Marathi Names
MARATHI_MALE = [
    "Aditya", "Ajinkya", "Aniket", "Chinmay", "Girish", "Hrishikesh",
    "Jayant", "Kedar", "Milind", "Ninad", "Omkar", "Prasad",
    "Rohan", "Sachin", "Tushar", "Vaibhav", "Yogesh", "Akshay",
    "Chaitanya", "Devendra", "Ganpat", "Hemant", "Jaydeep", "Kiran",
    "Mangesh", "Nitin", "Parag", "Rajan", "Saurabh", "Tanay",
]

MARATHI_FEMALE = [
    "Aarti", "Ashwini", "Gauri", "Janhavi", "Komal", "Madhura",
    "Prajakta", "Renuka", "Sanika", "Varsha", "Mugdha", "Sayali",
    "Tejal", "Snehal", "Manali", "Rutuja", "Sakshi", "Shraddha",
    "Tanuja", "Vidya", "Aparna", "Bhairavi", "Durga", "Harshada",
    "Juhi", "Ketaki", "Mrunalini", "Nutan", "Radhika", "Suvarna",
]

# Assorted pan-Indian names for more diversity
PAN_INDIAN = [
    "Vikram", "Akshara", "Shivangi", "Vedant", "Aarna", "Rehan",
    "Tanya", "Nirav", "Zoya", "Kabir", "Aarohi", "Veer",
    "Avni", "Advait", "Kirti", "Shlok", "Ahana", "Arush",
    "Inaaya", "Vivek", "Tanvi", "Ojas", "Kashvi", "Rishab",
    "Khushi", "Daksh", "Anvi", "Parv", "Ishika", "Yuvaan",
    "Mira", "Dev", "Sia", "Arnav", "Zara", "Shivansh",
    "Naina", "Hrithik", "Samaira", "Krish", "Advika", "Navin",
    "Mitali", "Aryan", "Shanaya", "Pranay", "Mahika", "Vivan",
    "Aadhira", "Ayush", "Radhika", "Pushkar", "Chaitra", "Hemang",
    "Gitanjali", "Nikhilesh", "Swaroop", "Chanakya", "Rukmini",
    "Satyam", "Sundari", "Acharya", "Achala", "Achyut", "Adhira",
    "Agastya", "Ahalya", "Ajay", "Akash", "Akshay", "Alok",
    "Amara", "Ambar", "Ambika", "Amrit", "Ananth", "Animesh",
    "Anurag", "Aparajita", "Apurva", "Ashwin", "Atman", "Avantika",
    "Bala", "Basant", "Bhanu", "Bhuvan", "Bindu", "Bodhi",
    "Chakor", "Champak", "Chandan", "Charan", "Charvi", "Charu",
    "Chetan", "Chitragupta", "Chitra", "Daksha", "Darsh", "Darpan",
    "Devaj", "Dhanya", "Dharini", "Dhruvi", "Eesha", "Eshan",
    "Falak", "Garv", "Gautami", "Girija", "Gowri", "Gul",
    "Hansa", "Hansika", "Harini", "Harit", "Hemani", "Hridaya",
    "Ila", "Inder", "Indrani", "Ipsita", "Ishan", "Ishwar",
    "Jagat", "Jahnavi", "Jalaj", "Janak", "Janardhan", "Jheel",
    "Jivika", "Jyotsna", "Kairav", "Kajal", "Kalash", "Kalyani",
    "Kamakshi", "Kanan", "Keshav", "Kimaya", "Kishore", "Kundali",
    "Lavanya", "Likhit", "Madhav", "Malhar", "Mantra", "Mayank",
    "Megha", "Mihika", "Milan", "Moksha", "Mrinal", "Mrudul",
    "Naksh", "Nalini", "Namrata", "Nandish", "Navin", "Neelam",
    "Niharika", "Niranjan", "Nishka", "Ojasvi",
    "Pallav", "Pankti", "Parinita", "Parineeti", "Parul", "Poorvi",
    "Pradnya", "Pragun", "Pranavi", "Pranjal", "Prarthana", "Pratham",
    "Prem", "Prisha", "Purav", "Purvi",
    "Rachit", "Raina", "Rajveer", "Rakhi", "Rani", "Rashika",
    "Ratna", "Revati", "Riddhima", "Rishi", "Rohan", "Rohini",
    "Rudransh", "Ruhi",
    "Sahas", "Sahana", "Sahil", "Saket", "Sameera", "Samhita",
    "Sankalp", "Saroj", "Sarvesh", "Satya", "Sharad", "Sharanya",
    "Shikha", "Shilpi", "Shivani", "Shravani", "Shravan", "Shreya",
    "Shriram", "Shubhangi", "Siddharth", "Snigdha", "Soham", "Somya",
    "Srishti", "Stuti", "Suhani", "Surya", "Swara", "Swarna",
    "Tahira", "Tamanna", "Tanishka", "Tanu", "Tapan", "Tapasya",
    "Tarini", "Tavish", "Tejaswi", "Trilok", "Trishna", "Tushar",
    "Uttara", "Ujjwal", "Urvi", "Urvashi",
    "Vaani", "Vaibhavi", "Vaishnavi", "Vanshika", "Varun", "Vedika",
    "Vibha", "Vihaan", "Vinay", "Vishal", "Vishnu", "Vivaan",
    "Yagna", "Yamini", "Yashvi", "Yuvan",
]

def generate_names():
    """Generate 1000 unique Indian names."""
    all_names = set()
    
    # Collect all names from all lists
    name_pools = [
        NORTH_MALE, NORTH_FEMALE,
        SOUTH_MALE, SOUTH_FEMALE,
        BENGALI_MALE, BENGALI_FEMALE,
        GUJARATI_MALE, GUJARATI_FEMALE,
        MARATHI_MALE, MARATHI_FEMALE,
        PAN_INDIAN,
    ]
    
    for pool in name_pools:
        for name in pool:
            all_names.add(name.strip())
    
    print(f"  Collected {len(all_names)} unique names from curated lists")
    
    # If we need more, generate compound/derived names
    # Common Indian name prefixes and suffixes
    prefixes = [
        "Anu", "Abhi", "Adi", "Aja", "Ama", "Ani", "Ari", "Ash",
        "Bal", "Bha", "Bhi", "Bra", "Cha", "Chi", "Dak", "Dha",
        "Dhi", "Dur", "Eka", "Gan", "Gau", "Gir", "Gu", "Har",
        "Hi", "Ish", "Ja", "Jag", "Jay", "Ka", "Kan", "Kar",
        "Ke", "Ki", "Ku", "La", "Ma", "Man", "Moh", "Mu",
        "Na", "Nar", "Nav", "Ni", "Om", "Pa", "Pra", "Pri",
        "Ra", "Raj", "Ram", "Ran", "Ri", "Ru", "Sa", "San",
        "Sar", "Sha", "Shi", "Shr", "Su", "Sur", "Swa", "Ta",
        "Tri", "Tu", "Uma", "Ut", "Va", "Ved", "Vi", "Vin",
        "Ya", "Yu",
    ]
    
    suffixes = [
        "av", "an", "ar", "ash", "esh", "ik", "il", "in", "ir",
        "it", "ith", "iv", "ya", "ani", "ika", "ini", "ita",
        "ati", "ali", "ari", "ashi", "esha", "ika", "ina",
        "ira", "isha", "ita", "ivi", "iya", "am", "um",
        "raj", "dev", "pal", "deep", "jeet", "nath", "kant",
        "anth", "esh", "endra", "inder", "ayan", "avan",
        "han", "van", "pan", "man", "tan", "ran", "san",
        "vin", "din", "hin", "shin", "dhar", "har", "kar",
    ]
    
    # Generate more names to reach 1000
    attempts = 0
    while len(all_names) < 1000 and attempts < 5000:
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        name = prefix + suffix
        # Capitalize properly
        name = name[0].upper() + name[1:]
        # Filter unreasonable names
        if 3 <= len(name) <= 14 and name not in all_names:
            all_names.add(name)
        attempts += 1
    
    # Convert to sorted list and take exactly 1000
    all_names = sorted(all_names)
    if len(all_names) > 1000:
        all_names = random.sample(all_names, 1000)
        all_names.sort()
    
    return all_names


def main():
    print("=" * 50)
    print("GENERATING INDIAN NAMES DATASET")
    print("=" * 50)
    
    names = generate_names()
    
    output_path = os.path.join(os.path.dirname(__file__), "TrainingNames.txt")
    with open(output_path, "w") as f:
        for name in names:
            f.write(name + "\n")
    
    print(f"\n  ✓ Generated {len(names)} unique Indian names")
    print(f"  ✓ Saved to: {output_path}")
    
    # Stats
    lengths = [len(n) for n in names]
    print(f"\n  Name length stats:")
    print(f"    Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}")
    print(f"    Sample names: {', '.join(random.sample(names, 10))}")


if __name__ == "__main__":
    main()
