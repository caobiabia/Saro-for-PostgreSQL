select  count(*) from comments as c,  		posts as p,           votes as v,  		badges as b,          users as u  where u.Id =c.UserId 	and c.UserId = p.OwnerUserId  	and p.OwnerUserId = v.UserId 	and v.UserId = b.UserId  AND c.Score=0  AND c.CreationDate>='2010-08-27 10:45:22'::timestamp  AND p.AnswerCount>=0  AND p.CommentCount<=12  AND p.FavoriteCount<=9  AND u.Reputation=6  AND u.CreationDate>='2010-07-28 11:39:00'::timestamp;