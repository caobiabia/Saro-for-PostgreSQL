select  count(*) from comments as c,  		posts as p,          postHistory as ph,          votes as v,          badges as b,          users as u  where u.Id = p.OwnerUserId     and u.Id = b.UserId     and p.Id = c.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND b.Date>='2010-08-04 03:10:29'::timestamp  AND c.Score=0  AND c.CreationDate<='2014-09-06 19:38:49'::timestamp  AND ph.CreationDate>='2011-04-20 14:01:49'::timestamp  AND p.Score<=96  AND p.ViewCount<=4847  AND p.AnswerCount<=4  AND p.CommentCount>=0  AND p.CommentCount<=9  AND u.Reputation>=1  AND u.Reputation<=329  AND u.Views=0  AND u.DownVotes>=0  AND u.DownVotes<=3;